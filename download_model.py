from sentence_transformers import SentenceTransformer

# Load the pretrained model from Hugging Face
model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Save it to your desired local folder
model.save("models/mpnet_contrastive_model_no_finetuning")
