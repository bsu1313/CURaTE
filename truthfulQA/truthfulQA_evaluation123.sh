
LORA_PATHS=(
  # "LTE_newinst2_HN_R"
  # "LTE_newinst2_R"
  # "LTE_newinst2_HN"
  # "LTE_newinst_HN_R_5epoch"
  # "NQ_LTU_21k_r16_a32_epoch5_wd0.01"
  "NQ_LTU_6k_r8_a16_epoch3_wd0.001"
  "NQ_LTU_9k_r8_a16_epoch3_wd0.001"
  "NQ_LTU_12k_r8_a16_epoch3_wd0.001"
  "NQ_LTU_15k_r8_a16_epoch3_wd0.001"
  # "output_LTE_newinst2_3HN"
)


for LORA_PATH in "${LORA_PATHS[@]}"; do
  deepspeed --include localhost:0 truthfulQA_evaluation123.py \
    --base_model "meta-llama/Llama-2-7b-chat-hf" \
    --lora_path "/home/work/hangyul/seyun_workspace/cache_LTE/$LORA_PATH" \
    --batch_size 32 \
    --ds_config "./ds_config.json" \
    --output_dir "./truthful_result_report_3/$LORA_PATH" \
    --custom_data_json "./truthfuQA_consent_false_only_augmented_llama_gen_consent_true_only.json"
done

