# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import copy
from dataclasses import dataclass, field
import json
import pathlib
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import datasets
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig,  prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class FineTunedModelArguments:
    qlora_model_name: Optional[str] = field(default="traditional_chinese_qlora_llama2")
    hf_api_key: Optional[str] = field(default="") # Change this to your own HF API key
    hf_username: Optional[str] = field(default="taiwan_llama2") # Change this to your own HF username

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="NousResearch/Llama-2-7b-chat-hf")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_cache_path: str = field(default=None,
                           metadata={"help": "Path to the cached jsonl data."})
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="paged_adamw_8bit")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    fdsp: bool = field(default=False)


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def last_index(lst, value):
    return next((len(lst) - i - 1 for i, x in enumerate(lst[::-1]) if x != value), -1)


def safe_ids(ids, max_value, pad_id):
    return [i if i < max_value else pad_id for i in ids]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
dummy_message = {
    "system": """\
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    "id": "dummy_message",
    "conversations": [
            {"from": "human", "value": "Who are you?"},
            {"from": "gpt", "value": "I am your virtual friend."},
            {"from": "human", "value": "What can you do?"},
            {"from": "gpt", "value": "I can chat with you."}
        ]
    }


def tokenize(item, tokenizer):
    roles = {"human": "user", "gpt": "assistant"}
    input_ids = []
    labels = []
    if "instruction" in item and len(item["instruction"]) > 0:
        system = item["instruction"]
    else:
        system = dummy_message["system"]
    system = B_SYS + system + E_SYS
    # add system before the first content in conversations
    item["conversations"][0]['value'] = system + item["conversations"][0]['value']
    for i, turn in enumerate(item["conversations"]):
        role = turn['from']
        content = turn['value']
        content = content.strip()
        if role == 'human':
            content = f"{B_INST} {content} {E_INST} "
            content_ids = tokenizer.encode(content)
            labels += [IGNORE_TOKEN_ID] * (len(content_ids))
        else:
            # assert role == "gpt"
            content = f"{content} "
            content_ids = tokenizer.encode(content, add_special_tokens=False) + [tokenizer.eos_token_id]   # add_special_tokens=False remove bos token, and add eos at the end
            labels += content_ids
        input_ids += content_ids

    input_ids = input_ids[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]

    trunc_id = last_index(labels, IGNORE_TOKEN_ID) + 1
    input_ids = input_ids[:trunc_id]
    labels = labels[:trunc_id]
    if len(labels) == 0:
        return tokenize(dummy_message, tokenizer)
    input_ids = safe_ids(input_ids, tokenizer.vocab_size, tokenizer.pad_token_id)
    labels = safe_ids(labels, tokenizer.vocab_size, IGNORE_TOKEN_ID)
    return input_ids, labels


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        item = self.raw_data[i]
        input_ids, labels = tokenize(
            copy.deepcopy(item),
            self.tokenizer)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        ret = dict(
            input_ids=input_ids,
            labels=labels,
        )
        self.cached_data_dict[i] = ret

        return ret


def load_json(data_path, data_cache_path):
    if data_path.endswith(".json"):
        return json.load(open(data_path, "r"))
    elif data_path.endswith(".jsonl"):
        dataset = datasets.load_dataset('json', data_files=data_path, cache_dir=data_cache_path)
        return dataset['train']
    elif os.path.exists(data_path):
        dataset = datasets.load_from_disk(data_path)
        return dataset['train']
    else:
        dataset = datasets.load_dataset(data_path, cache_dir=data_cache_path)
        return dataset['train']


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_TOKEN_ID)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    rank0_print("Loading data...")

    train_json = load_json(data_args.data_path, data_args.data_cache_path)
    train_dataset = LazySupervisedDataset(train_json, tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


#class CustomTrainer(Trainer):
class CustomTrainer(SFTTrainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if self.fsdp is not None:
            if output_dir is None:
                output_dir = self.args.output_dir
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                FullStateDictConfig,
                StateDictType,
            )
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = self.model.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=cpu_state_dict)  # noqa
            # Push to the Hub when `save_model` is called by the user.
            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(commit_message="Model save")
        else:
            super().save_model(output_dir, _internal_call)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (FineTunedModelArguments, ModelArguments, DataArguments, TrainingArguments)
    )
    finetuned_model_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Create directories to store final qlora fine-tuned model
    script_dir = Path(__file__).parent.absolute()
    project_dir = script_dir.parent.absolute()
    default_qlora_model_path = os.path.join(project_dir, finetuned_model_args.qlora_model_name)
    os.makedirs(default_qlora_model_path, exist_ok=True)

    # login to your huggingface account with your API key for push qlora model to your huggingface accoun
    try:
        # Get the API token from the finetuned_model_args
        hf_api_key = finetuned_model_args.hf_api_key
        # Login to the Hugging Face Hub
        login(token=hf_api_key)
    except Exception as e:
        print(e)
        print("Please set the HF_API_KEY at train.sh to push the model to the Hugging Face Hub.")

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
    )


    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = CustomTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=training_args.model_max_length,
        **data_module
    )

    if training_args.deepspeed or training_args.gradient_checkpointing:
        model.config.use_cache = False
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    print("Ready to save model")
    model.save_pretrained(default_qlora_model_path)
    print("Model saved to", default_qlora_model_path)

    # Push to the huggingace Hub
    try:
        default_huggingface_repo = f"{finetuned_model_args.hf_username}/{finetuned_model_args.qlora_model_name}"
        model.push_to_hub(default_huggingface_repo , use_auth_token=True)
    except Exception as e:
        print(e)
        print("Please set the HF_API_KEY/HF_USERNAME at train.sh to push the model to the Hugging Face Hub.")


if __name__ == "__main__":
    train()