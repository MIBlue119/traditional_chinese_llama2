"""Scripts to merge a fine-tuned model with the original model.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import transformers
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

@dataclass
class MergedModelArguments:
    output_merged_dir: Optional[str] = field(default="./traditional_chinese_qlora_llama2_merged")

@dataclass
class FineTunedModelArguments:
    qlora_model_name: Optional[str] = field(default="traditional_chinese_qlora_llama2")
    hf_username: Optional[str] = field(default="weiren119") # Change this to your own HF username

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="NousResearch/Llama-2-7b-chat-hf")



parser = transformers.HfArgumentParser(
        (MergedModelArguments, FineTunedModelArguments, ModelArguments)
)

merged_model_args, finetuned_model_args, model_args = parser.parse_args_into_dataclasses()

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
# Merge the adapted model with the original model
model = model.merge_and_unload()

output_merged_dir = merged_model_args.output_merged_dir
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy use
tokenizer.save_pretrained(output_merged_dir)