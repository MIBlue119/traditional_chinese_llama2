# Script to merge the pretrained model with the qlora model

# Specify the path to save the merged model
OUTPUT_MERGED_DIR="./traditional_chinese_qlora_llama2_merged"

# We would pull the qlora model from huggingface, so you need to fill the following two variables
# the model repo name would be ${HF_USERNAME}/${QLORA_MODEL_NAME}
# Fill with your qlora model name
QLORA_MODEL_NAME="traditional_chinese_qlora_llama2"
# Fill with your huggingface username
HF_USERNAME="weiren119"

# Fill with your desired pretrained model souce repo name
# Currently i use the repo from NousResearch: https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
MODEL_PATH="NousResearch/Llama-2-7b-chat-hf" 

poetry run python merge.py \
        --output_merged_dir ${OUTPUT_MERGED_DIR} \
        --qlora_model_name ${QLORA_MODEL_NAME} \
        --hf_username ${HF_USERNAME} \
        --model_name_or_path ${MODEL_PATH}