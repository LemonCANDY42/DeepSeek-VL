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

# Load models
def load_models():
    models = {
        "DeepSeek-VL 7B": "/mnt/scsi_disk/deepseek-vl-7b-chat",
    }
    for model_name in models:
        models[model_name] = load_model(models[model_name])
    return models

models = load_models()

def generate_prompt_with_history(text, image, history, vl_chat_processor, tokenizer, max_length=2048):
    sft_format = "deepseek"
    user_role_ind = 0
    bot_role_ind = 1

    conversation = vl_chat_processor.new_chat_template()

    if history:
        conversation.messages = history

    if image is not None:
        if "<image_placeholder>" not in text:
            text = "<image_placeholder>" + "\n" + text
        text = (text, image)

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

@app.route('/chat', methods=['POST'])
def chat():
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

    if image_data:
        image = Image.open(BytesIO(base64.b64decode(image_data)))
    else:
        image = None

    try:
        tokenizer, vl_gpt, vl_chat_processor = models[model_name]
    except KeyError:
        return jsonify({'error': 'Model not found'}), 400

    conversation = generate_prompt_with_history(
        text,
        image,
        history,
        vl_chat_processor,
        tokenizer,
        max_length=max_context_length_tokens,
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

    return jsonify({
        'response': response,
        'history': conversation.messages
    })

@app.route('/chat_no_history', methods=['POST'])
def chat_no_history():
    data = request.json
    text = data.get('text')
    image_data = data.get('image')
    model_name = data.get('model', 'DeepSeek-VL 7B')
    top_p = data.get('top_p', 0.95)
    temperature = data.get('temperature', 0.1)
    repetition_penalty = data.get('repetition_penalty', 1.1)
    max_length_tokens = data.get('max_length_tokens', 2048)

    if image_data:
        image = Image.open(BytesIO(base64.b64decode(image_data)))
    else:
        image = None

    try:
        tokenizer, vl_gpt, vl_chat_processor = models[model_name]
    except KeyError:
        return jsonify({'error': 'Model not found'}), 400

    conversation = vl_chat_processor.new_chat_template()
    if image is not None:
        if "<image_placeholder>" not in text:
            text = "<image_placeholder>" + "\n" + text
        text = (text, image)

    conversation.append_message(conversation.roles[0], text)
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

    return jsonify({
        'response': response
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8222)
