IGNORE_TOKEN_ID = -100
IMAGE_TOKEN_ID = -200
IMAGE_TOKEN = "<image>"
IM_START = "<|im_start|>"
IM_SEP = "<|im_sep|>"
IM_END = "<|im_end|>"
SYSTEM_PROMPT = (
    "You are a medieval knight and must provide explanations to modern people."
)
# CHAT_TEMPLATE_WITHOUT_SYSTEM_MESSAGE = "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>\n' + message['content'] + '<|im_end|><|im_start|>assistant<|im_sep|>'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}"
CHAT_TEMPLATE_WITHOUT_SYSTEM_MESSAGE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '<|im_sep|>' + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant<|im_sep|>\n'}}{% endif %}"
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "openai": [0.48145466, 0.4578275, 0.40821073],
    "inception": [0.5, 0.5, 0.5],
}
STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "openai": [0.26862954, 0.26130258, 0.27577711],
    "inception": [0.5, 0.5, 0.5],
}
