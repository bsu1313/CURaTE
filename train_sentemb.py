from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import util
from sentence_transformers.losses import SiameseDistanceMetric
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import sys


# 1. Load JSON and prepare InputExamples
def load_input_examples(json_path):
    examples = []

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

        for item in raw:
            human_text = ""
            for turn in item["conversations"]:
                if turn["from"] == "human":
                    full_text = turn["value"]
                    if "\n\n" in full_text:
                        # Split at first double newline
                        human_text = full_text.split("\n\n", 1)[1].strip()
                    else:
                        # Fallback: use the whole text
                        human_text = full_text.strip()

            # Extract sentences
            sent_A, sent_B = None, None
            try:
                parts = human_text.split("[Query]:")
                forgotten_part = parts[0].replace("[Forgotten Information]:", "").strip()
                query_part = parts[1].strip()
                sent_A = forgotten_part
                sent_B = query_part
            except Exception as e:
                print("⚠️ Could not parse text:", human_text)
                continue

            # Label
            label = 1.0 if item["features"] == "A'" else 0.0
            # print(f"Processing: {sent_A} | {sent_B} | Label: {label}")
            # sys.exit()

            # Create InputExample
            example = InputExample(
                texts=[sent_A, sent_B],
                label=label
            )
            examples.append(example)

    return examples


# 2. Main training function
def main(
    json_path,
    output_path="mpnet_contrastive_model",
    epochs=1,
    batch_size=16,
    learning_rate=2e-5
):
    # Load data
    print("✅ Loading data...")
    examples = load_input_examples(json_path)
    print(f"Prepared {len(examples)} pairs")

    # Load model
    print("✅ Loading pre-trained model...")
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # model = SentenceTransformer('bert-base-nli-mean-tokens')
    # model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    # model = SentenceTransformer('sentence-transformers-testing/stsb-bert-tiny-safetensors')

    # DataLoader
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    # ContrastiveLoss
    train_loss = losses.ContrastiveLoss(
        model=model,
        distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
        # distance_metric=SiameseDistanceMetric.EUCLIDEAN_DISTANCE
        margin=0.5
    )

    # Train
    print("✅ Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True,
        output_path=output_path
    )
    print(f"✅ Model saved to {output_path}")


if __name__ == "__main__":
    # main(
    #     json_path="NQ_LTU_18k.json",
    #     epochs=1,
    #     batch_size=16,
    #     learning_rate=2e-5
    # )

    ablation_files = [
        "NQ_CURE_12K_a",
        "NQ_CURE_18K_a",
        "NQ_CURE_18K_a_no_b",
        "NQ_CURE_NO_HN_18K_a",
        "NQ_CURE_NO_HN_18K_a_no_b"
        "TQ_CURE_18K_a",
        ]

    for file in ablation_files:
        main(
            json_path=f"ablation/{file}.json",
            output_path=f"models/mpnet_contrastive_model_{file}",
            epochs=1,
            batch_size=16,
            learning_rate=2e-5
        )
