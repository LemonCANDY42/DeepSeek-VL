import base64
import json
from flask import Flask, request, jsonify
from io import BytesIO
import torch
from PIL import Image
from deepseek_vl.serve.inference import (
    convert_conversation_to_prompts,
    deepseek_generate,
    load_model,
)
from deepseek_vl.utils.conversation import SeparatorStyle

app = Flask(__name__)

# 全局字典，用于存储每个 IP 地址的对话历史
conversation_history = {}
# 用于存储图像数据
image_storage = {}

# Load models
def load_models():
    models = {
        "DeepSeek-VL 7B": "/mnt/scsi_disk/deepseek-vl-7b-chat",
    }
    for model_name in models:
        models[model_name] = load_model(models[model_name])
    return models

models = load_models()
"""
    system_prompt = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )
"""
DEFAULT_SYSTEM_PROMPT = (
        "你是语言和视觉方面的得力助手。"
        "你能理解用户提供的视觉内容"
        "并用自然语言协助用户完成实时的构图指导，并用中文回复"
    )

RE_TAKE_PHOTO_SYSTEM_PROMPT = (
        "你是语言和视觉方面的得力助手。"
        "你能理解用户提供的视觉内容"
        "并用自然语言协助用户实时的构图指导，并用中文回复"
        "你可以关注到之前提到的建议和图像，如果用户新的图像有所改善，适当切稍微的鼓励和提高审美评分"
    )


"""
    你作为一个摄影审美和构图的专家，会按照0-10的分数给我提供的正在实时拍摄的图片进行实事求是且合理的审美评分（0：极差，3：较差，5：一般，7：较好，9：优秀）。请从以下几个维度进行综合评价：
    构图、光线、色彩、清晰度、情感表达。并根据画面的好坏给出具体的构图建议（如：移除某元素，将画面向某处移动，改变拍摄角度等）。建议需具体且可操作，字数不超过150字。格式如下：\n审美评分:53\n构图建议:\nxxxx
"""

def generate_prompt_with_history(text, image, history, vl_chat_processor, tokenizer, max_length=2048, image_counter=1):
    sft_format = "deepseek"
    user_role_ind = 0
    bot_role_ind = 1

    conversation = vl_chat_processor.new_chat_template()

    if history:
        # 替换历史记录中的图像数据为带有顺序编号的占位符
        for role, message in history:
            if isinstance(message, tuple) and message[1].startswith("<image_placeholder"):
                conversation.append_message(role, (message[0], f"<image_placeholder_{image_counter}>"))
                image_counter += 1
            else:
                conversation.append_message(role, message)

    if image is not None:
        if "<image_placeholder>" not in text:
            text = f"<image_placeholder_{image_counter}>" + "\n" + text
        text = (text, image)  # 使用实际图像数据

    conversation.append_message(conversation.roles[user_role_ind], text)
    conversation.append_message(conversation.roles[bot_role_ind], "")

    conversation_copy = conversation.copy()

    rounds = len(conversation.messages) // 2

    for _ in range(rounds):
        current_prompt = get_prompt(conversation)
        current_prompt = current_prompt.replace("</s>", "") if sft_format == "deepseek" else current_prompt

        if torch.tensor(tokenizer.encode(current_prompt)).size(-1) <= max_length:
            return conversation_copy

        if len(conversation.messages) % 2 != 0:
            return None

        try:
            for _ in range(2):
                conversation.messages.pop(0)
        except IndexError:
            return None

    return None


def get_prompt(conv) -> str:
    system_prompt = conv.system_template.format(system_message=conv.system_message)
    if conv.sep_style == SeparatorStyle.DeepSeek:
        seps = [conv.sep, conv.sep2]
        if system_prompt == "" or system_prompt is None:
            ret = ""
        else:
            ret = system_prompt + seps[0]
        for i, (role, message) in enumerate(conv.messages):
            if message:
                if type(message) is tuple:
                    message, _ = message
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret
    else:
        return conv.get_prompt
    
def correct_base64_padding(base64_str):
    return base64_str + '=' * (-len(base64_str) % 4)


@app.route('/chat', methods=['POST'])
def chat():
    image_counter = 1
    
    data = request.json
    text = data.get('text')
    image_data = data.get('image')
    history = data.get('history', [])
    model_name = data.get('model', 'DeepSeek-VL 7B')
    top_p = data.get('top_p', 0.95)
    temperature = data.get('temperature', 0.1)
    repetition_penalty = data.get('repetition_penalty', 1.1)
    max_length_tokens = data.get('max_length_tokens', 2048)
    max_context_length_tokens = data.get('max_context_length_tokens', 4096)
    
    # 获取用户的 IP 地址
    user_ip = request.remote_addr

    # 获取或创建该 IP 地址的对话历史
    if user_ip not in conversation_history:
        conversation_history[user_ip] = {'history': [], 'image_counter': 1}

    # 当前对话历史
    current_history = conversation_history[user_ip]


    if image_data:
        # Remove the data URL prefix if it exists
        if image_data.startswith('data:image'):
            image_data = image_data.split(',', 1)[1]
        image_data = correct_base64_padding(image_data)
        try:
            decoded_image_data = base64.b64decode(image_data)
            image = Image.open(BytesIO(decoded_image_data))
            
            # Convert RGBA (4 channels) to RGB (3 channels)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                
        except (base64.binascii.Error, PIL.UnidentifiedImageError) as e:
            print(f"Error decoding image: {e}")
            print(f"Base64 data (first 100 chars): {image_data[:100]}")
            return jsonify({'error': 'Invalid image data'}), 400
    else:
        image = None

    try:
        tokenizer, vl_gpt, vl_chat_processor = models[model_name]
    except KeyError:
        return jsonify({'error': 'Model not found'}), 400
    
    
    conversation = generate_prompt_with_history(
        text,
        image,
        current_history,  # 使用当前用户的对话历史
        vl_chat_processor,
        tokenizer,
        max_length=max_context_length_tokens,
        image_counter=image_counter
    )
    if conversation is None:
        return jsonify({'error': 'Failed to generate prompt with history'}), 400

    prompts = convert_conversation_to_prompts(conversation)
    stop_words = conversation.stop_str

    full_response = ""
    with torch.no_grad():
        for x in deepseek_generate(
            prompts=prompts,
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            max_length=max_length_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
        ):
            full_response += x

    response = full_response.strip()
    conversation.update_last_message(response)
    
    conversation.set_system_message(RE_TAKE_PHOTO_SYSTEM_PROMPT)
    
    # 更新对话历史，排除不可序列化的对象
    serializable_messages = []
    for role, message in conversation.messages:
        if isinstance(message, tuple) and isinstance(message[1], Image.Image):
            serializable_messages.append((role, (message[0], f"<image_placeholder_{image_counter}>")))
            image_counter += 1
        else:
            serializable_messages.append((role, message))

    conversation_history[user_ip]['history'] = serializable_messages
    conversation_history[user_ip]['image_counter'] = image_counter

    return jsonify({
        'response': response,
        'history': conversation_history[user_ip]
    })

@app.route('/chat_no_history', methods=['POST'])
def chat_no_history():
    data = request.json
    user_text = data.get('text')
    image_data = data.get('image')
    model_name = data.get('model', 'DeepSeek-VL 7B')
    top_p = data.get('top_p', 0.95)
    temperature = data.get('temperature', 0.1)
    repetition_penalty = data.get('repetition_penalty', 1.1)
    max_length_tokens = data.get('max_length_tokens', 2048)

    # Add default system prompt
    system_text = DEFAULT_SYSTEM_PROMPT

    if image_data:
        # Remove the data URL prefix if it exists
        if image_data.startswith('data:image'):
            image_data = image_data.split(',', 1)[1]
        image_data = correct_base64_padding(image_data)
        try:
            decoded_image_data = base64.b64decode(image_data)
            image = Image.open(BytesIO(decoded_image_data))
            
            # Convert RGBA (4 channels) to RGB (3 channels)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                
        except (base64.binascii.Error, PIL.UnidentifiedImageError) as e:
            print(f"Error decoding image: {e}")
            print(f"Base64 data (first 100 chars): {image_data[:100]}")
            return jsonify({'error': 'Invalid image data'}), 400
    else:
        image = None

    try:
        tokenizer, vl_gpt, vl_chat_processor = models[model_name]
    except KeyError:
        return jsonify({'error': 'Model not found'}), 400

    conversation = vl_chat_processor.new_chat_template()
    if image is not None:
        if "<image_placeholder>" not in user_text:
            user_text = "<image_placeholder>" + "\n" + user_text
        user_text = (user_text, image)
        
    conversation.set_system_message(DEFAULT_SYSTEM_PROMPT)

    conversation.append_message(conversation.roles[1], user_text)
    conversation.append_message(conversation.roles[1], "")

    prompts = convert_conversation_to_prompts(conversation)
    stop_words = conversation.stop_str

    full_response = ""
    with torch.no_grad():
        for x in deepseek_generate(
            prompts=prompts,
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            max_length=max_length_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
        ):
            full_response += x

    response = full_response.strip()
    conversation.update_last_message(response)
    
    print(f"")

    return jsonify({
        'response': response
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8222)

"""
    curl -X POST http://localhost:8222/chat_no_history \
        -H "Content-Type: application/json" \
        -d '{
            "text": "This is a system prompt\nWhat is this image about?",
            "image": "<BASE64_IMAGE_STRING>",
            "model": "DeepSeek-VL 7B",
            "top_p": 0.95,
            "temperature": 0.1,
            "repetition_penalty": 1.1,
            "max_length_tokens": 2048
        }'
"""