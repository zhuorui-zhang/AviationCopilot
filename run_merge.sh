# python merge_peft_adapter.py \
#   --model_type auto \
#   --base_model meta-llama/Llama-3.1-8B-Instruct  \
#   --tokenizer_path meta-llama/Llama-3.1-8B-Instruct  \
#   --lora_model outputs-pt-llama3.1-8B-Instruct \
#   --output_dir outputs-pt-llama3.1-8B-Instruct-merged


python merge_peft_adapter.py \
  --model_type auto \
  --base_model /media/zhangzr/4TB/hf_models/Llama-3.1-8B  \
  --tokenizer_path /media/zhangzr/4TB/hf_models/Llama-3.1-8B  \
  --lora_model /media/zhangzr/4TB/vcopilot/outputs-pt-2 \
  --output_dir outputs-pt-llama3.1-8B-merged-2