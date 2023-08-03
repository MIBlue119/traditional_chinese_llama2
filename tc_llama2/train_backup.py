from random import randrange

from datasets import load_dataset
from loguru import logger
from peft import LoraConfig,  prepare_model_for_kbit_training, get_peft_model
import torch
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments


from tc_llama2.settings import app_settings

logger.info(torch.cuda.get_device_capability()[0])
logger.info(app_settings)


def format_instruction(sample):
    """Format a sample from the dataset into a markdown string.
    
    ref:https://www.philschmid.de/instruction-tune-llama-2
    """
    logger.info(sample)
    f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
{sample['input']}

### Response:
{sample['instruction']}
"""

if app_settings.FINETUNED_DATA_PATH_LIST == []:
    logger.info("No data paths found. Please set the environment variable FINETUNED_DATA_PATH_LIST.")
    exit(1)

for data_path in app_settings.FINETUNED_DATA_PATH_LIST:
    dataset = load_dataset("json", data_files=data_path,split="train")
    logger.info(dataset)
    logger.info(format_instruction(dataset[randrange(len(dataset))]))


use_flash_attention = False
# COMMENT IN TO USE FLASH ATTENTION
# replace attention with flash attention
# if torch.cuda.get_device_capability()[0] >= 8:
#     from utils.llama_patch import replace_attn_with_flash_attn
#     print("Using flash attention")
#     replace_attn_with_flash_attn()
#     use_flash_attention = True

# Hugging Face model id
model_id = "NousResearch/Llama-2-7b-hf" # non-gated


# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache=False, device_map="auto")
model.config.pretraining_tp = 1

# Validate that the model is using flash attention, by comparing doc strings
if use_flash_attention:
    from utils.llama_patch import forward
    assert model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__, "Model is not using flash attention"


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


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

args = TrainingArguments(
    output_dir="llama-7-int4-alpaca",
    num_train_epochs=3,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True # disable tqdm since with packing values are in correct
)

max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)


# train
trainer.train()

# save model
trainer.save_model()
logger.info("Done.")