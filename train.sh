DATASET="/mnt/HDD2/llama2/traditional_chinese_llama2/data/alpaca-tw_en-align.json"

DATA_CACHE_PATH="hf_datasets_cache"
MODEL_PATH="NousResearch/Llama-2-7b-hf"

output_dir="./checkpoints_llama2"

# torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
#     --master_port=25003 \
#         tc_llama2/train.py \
#         --model_name_or_path ${MODEL_PATH} \
#         --data_path ${DATASET} \
#         --data_cache_path ${DATA_CACHE_PATH} \
#         --bf16 True \
#         --output_dir ${output_dir} \
#         --num_train_epochs 1 \
#         --per_device_train_batch_size 4 \
#         --per_device_eval_batch_size 4 \
#         --gradient_accumulation_steps 1 \
#         --evaluation_strategy 'no' \
#         --save_strategy 'steps' \
#         --save_steps 1200 \
#         --save_total_limit 5 \
#         --learning_rate 2e-5 \
#         --weight_decay 0. \
#         --warmup_ratio 0.03 \
#         --lr_scheduler_type cosine \
#         --logging_steps 1 \
#         # --fsdp 'full_shard auto_wrap' \
#         # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#         --tf32 True \
#         --model_max_length 4096 \
#         --gradient_checkpointing True


poetry run python tc_llama2/train.py \
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
