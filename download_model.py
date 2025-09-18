# from sentence_transformers import SentenceTransformer

# # Load the pretrained model from Hugging Face
# model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

# # Save it to your desired local folder
# model.save("models/mpnet_contrastive_model_no_finetuning")


from huggingface_hub import snapshot_download
import os
import sys

# Destination folder
# model_path = "models/llama3.2_base_scienceqa"
# model_path = "models/O3_LLAMA2_ScienceQA"
# model_path = "models/Llama-3.2-1B-Instruct"
model_path = "models/Llama-2-7b-chat-hf"
os.makedirs(model_path, exist_ok=True)


# Download model from HF
# snapshot_download(repo_id="laurel1313/llama3.2_base_scienceqa", local_dir=model_path)
# snapshot_download(repo_id="gcyzsl/O3_LLAMA2_ScienceQA", local_dir=model_path)
# snapshot_download(repo_id="meta-llama/Llama-3.2-1B-Instruct", local_dir=model_path)
snapshot_download(repo_id="meta-llama/Llama-2-7b-chat-hf", local_dir=model_path)