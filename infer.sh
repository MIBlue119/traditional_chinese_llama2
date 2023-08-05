# Fill with your desired pretrained model souce repo name
# Currently i use the repo from NousResearch: https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
MODEL_PATH="NousResearch/Llama-2-7b-chat-hf" 

# Fill with your desired qlora model name
QLORA_MODEL_NAME="traditional_chinese_qlora_llama2"
# If you want to push the model to huggingface, you need to fill the following two variables
# Get your huggingface api key from https://huggingface.co/docs/hub/security-tokens
# Otherwise, you can leave them empty

# Fill with your huggingface api key
HF_API_KEY="None"
# Fill with your huggingface username
HF_USERNAME="weiren119"

poetry run python infer.py \
        --qlora_model_name ${QLORA_MODEL_NAME} \
        --hf_api_key ${HF_API_KEY} \
        --hf_username ${HF_USERNAME} \
        --model_name_or_path ${MODEL_PATH}