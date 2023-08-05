"""
Modified from https://github.com/LinkSoul-AI/Chinese-Llama-2-7b/blob/main/infer.py

Support load qLora model from HuggingFace model hub
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import transformers
import torch
from transformers import AutoTokenizer, TextStreamer
from peft import AutoPeftModelForCausalLM

@dataclass
class FineTunedModelArguments:
    qlora_model_name: Optional[str] = field(default="traditional_chinese_qlora_llama2")
    hf_api_key: Optional[str] = field(default="") # Change this to your own HF API key
    hf_username: Optional[str] = field(default="weiren119") # Change this to your own HF username

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="NousResearch/Llama-2-7b-chat-hf")



parser = transformers.HfArgumentParser(
        (FineTunedModelArguments, ModelArguments)
)

finetuned_model_args, model_args = parser.parse_args_into_dataclasses()

# We use the tokenizer from the original model
original_model_path=model_args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(original_model_path, use_fast=False)

# We load our fine-tuned model, you can replace this with your own model
qlora_model_path = f"{finetuned_model_args.hf_username}/{finetuned_model_args.qlora_model_name}"
model = AutoPeftModelForCausalLM.from_pretrained(
        qlora_model_path,
        load_in_4bit=qlora_model_path.endswith("4bit"),
        torch_dtype=torch.float16,
        device_map='auto'
    )
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# You could generate from https://patorjk.com/software/taag/#p=display&h=0&f=Big%20Money-ne&t=Taiwan%20LLama2%0ATaiwan%20number%20one
model_log = """\

 /$$$$$$$$           /$$                                         /$$       /$$                                          /$$$$$$                                     
|__  $$__/          |__/                                        | $$      | $$                                         /$$__  $$                                    
   | $$     /$$$$$$  /$$ /$$  /$$  /$$  /$$$$$$  /$$$$$$$       | $$      | $$        /$$$$$$  /$$$$$$/$$$$   /$$$$$$ |__/  \ $$                                    
   | $$    |____  $$| $$| $$ | $$ | $$ |____  $$| $$__  $$      | $$      | $$       |____  $$| $$_  $$_  $$ |____  $$  /$$$$$$/                                    
   | $$     /$$$$$$$| $$| $$ | $$ | $$  /$$$$$$$| $$  \ $$      | $$      | $$        /$$$$$$$| $$ \ $$ \ $$  /$$$$$$$ /$$____/                                     
   | $$    /$$__  $$| $$| $$ | $$ | $$ /$$__  $$| $$  | $$      | $$      | $$       /$$__  $$| $$ | $$ | $$ /$$__  $$| $$                                          
   | $$   |  $$$$$$$| $$|  $$$$$/$$$$/|  $$$$$$$| $$  | $$      | $$$$$$$$| $$$$$$$$|  $$$$$$$| $$ | $$ | $$|  $$$$$$$| $$$$$$$$                                    
   |__/    \_______/|__/ \_____/\___/  \_______/|__/  |__/      |________/|________/ \_______/|__/ |__/ |__/ \_______/|________/   
                                                                                     """

logo = """\
 /$$$$$$$$           /$$                                                                           /$$                                                              
|__  $$__/          |__/                                                                          | $$                                                              
   | $$     /$$$$$$  /$$ /$$  /$$  /$$  /$$$$$$  /$$$$$$$        /$$$$$$$  /$$   /$$ /$$$$$$/$$$$ | $$$$$$$   /$$$$$$   /$$$$$$         /$$$$$$  /$$$$$$$   /$$$$$$ 
   | $$    |____  $$| $$| $$ | $$ | $$ |____  $$| $$__  $$      | $$__  $$| $$  | $$| $$_  $$_  $$| $$__  $$ /$$__  $$ /$$__  $$       /$$__  $$| $$__  $$ /$$__  $$
   | $$     /$$$$$$$| $$| $$ | $$ | $$  /$$$$$$$| $$  \ $$      | $$  \ $$| $$  | $$| $$ \ $$ \ $$| $$  \ $$| $$$$$$$$| $$  \__/      | $$  \ $$| $$  \ $$| $$$$$$$$
   | $$    /$$__  $$| $$| $$ | $$ | $$ /$$__  $$| $$  | $$      | $$  | $$| $$  | $$| $$ | $$ | $$| $$  | $$| $$_____/| $$            | $$  | $$| $$  | $$| $$_____/
   | $$   |  $$$$$$$| $$|  $$$$$/$$$$/|  $$$$$$$| $$  | $$      | $$  | $$|  $$$$$$/| $$ | $$ | $$| $$$$$$$/|  $$$$$$$| $$            |  $$$$$$/| $$  | $$|  $$$$$$$
   |__/    \_______/|__/ \_____/\___/  \_______/|__/  |__/      |__/  |__/ \______/ |__/ |__/ |__/|_______/  \_______/|__/             \______/ |__/  |__/ \_______/
                                                         """



def get_prompt(message: str, chat_history: list[tuple[str, str]]) -> str:
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in chat_history:
        texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{message.strip()} [/INST]')
    return ''.join(texts)


print ("="*100)
print (model_log)
print (logo)
print ("-"*80)
print ("Have a try!")

s = ''
chat_history = []
while True:
    s = input("User: ")
    if s != '':
        prompt = get_prompt(s, chat_history)
        print ('Answer:')
        tokens = tokenizer(prompt, return_tensors='pt').input_ids
        #generate_ids = model.generate(tokens.cuda(), max_new_tokens=4096, streamer=streamer)
        generate_ids = model.generate(input_ids=tokens.cuda(), max_new_tokens=4096, streamer=streamer)
        output = tokenizer.decode(generate_ids[0, len(tokens[0]):-1]).strip()
        chat_history.append([s, output])
        print ('-'*80)