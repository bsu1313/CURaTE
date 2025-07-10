
LORA_PATHS=(
  # "LTE_newinst2_HN_R"
  # "LTE_newinst2_R"
  # "LTE_newinst2_HN"
  # "LTE_newinst_HN_R_5epoch"
  "LTE_newinst5"
  # "output_LTE_newinst2_3HN"
)


for LORA_PATH in "${LORA_PATHS[@]}"; do
  deepspeed --include localhost:0 evaluation_commonsenseQA.py \
    --base_model "meta-llama/Llama-2-7b-chat-hf" \
    --lora_path "/home/work/hangyul/seyun_workspace/cache_LTE/$LORA_PATH" \
    --batch_size 32 \
    --ds_config "./ds_config.json" \
    --output_dir "./truthful_commonsense_result/$LORA_PATH" \
    --id_mapping_json "./csqa_to_truthqa_top3.json"
done

