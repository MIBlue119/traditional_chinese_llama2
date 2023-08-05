
# Your supervised finetuning dataset path
DATASET="./data/alpaca-tw_en-align_converted.json"
DATA_CACHE_PATH="hf_datasets_cache"
# Fill with your desired pretrained model souce repo name
# Currently i use the repo from NousResearch: https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
MODEL_PATH="NousResearch/Llama-2-7b-chat-hf" 
# Fill the path to save the finetuned model's checkpoints
output_dir="./checkpoints_llama2"

# Fill with your desired qlora model name
QLORA_MODEL_NAME="traditional_chinese_qlora_llama2"
# If you want to push the model to huggingface, you need to fill the following two variables
# Get your huggingface api key from https://huggingface.co/docs/hub/security-tokens
# Otherwise, you can leave them empty

# Fill with your huggingface api key
HF_API_KEY="None"
# Fill with your huggingface username
HF_USERNAME="None"

poetry run python tc_llama2/train.py \
        --qlora_model_name ${QLORA_MODEL_NAME} \
        --hf_api_key ${HF_API_KEY} \
        --hf_username ${HF_USERNAME} \
        --model_name_or_path ${MODEL_PATH} \
        --data_path ${DATASET} \
        --data_cache_path ${DATA_CACHE_PATH} \
        --bf16 True \
        --output_dir ${output_dir} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy 'no' \
        --save_strategy 'steps' \
        --save_steps 1200 \
        --save_total_limit 5 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 4096 \
        --gradient_checkpointing True
