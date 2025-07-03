import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import sys

# 1. Load JSON and split
def load_and_split_data(json_path, test_size=0.2, random_state=42):
    data = []
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

            label = 1 if item["features"] == "A'" else 0

            data.append({
                "text": human_text,
                "label": label
            })
        # print("data sample:", data[0])

    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    return train_data, test_data

# 2. Dataset class
class ConversationDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }

# 3. Training function
def train(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in progress:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # print("decoded input_ids:", input_ids[0])
        # print("labels:", labels[0])
        # sys.exit()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} average loss: {avg_loss:.4f}")

# 4. Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute classification report
    report = classification_report(
        all_labels, all_preds, target_names=["not A'", "A'"]
    )
    print("\n✅ Evaluation Report:\n")
    print(report)

# 5. Main function
def main(
    json_path,
    epochs=3,
    batch_size=8,
    lr=2e-5,
    test_size=0.2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and split
    train_data, test_data = load_and_split_data(json_path, test_size=test_size)
    print(f"✅ Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = ConversationDataset(train_data, tokenizer)
    test_dataset = ConversationDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        train(model, train_loader, optimizer, device, epoch)

    # Save model
    model.save_pretrained("roberta_features_Aprime_classifier")
    tokenizer.save_pretrained("roberta_features_Aprime_classifier")

    # Evaluate
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    # Example usage
    main(
        json_path="NQ_LTU_18k.json",
        epochs=3,
        batch_size=8,
        lr=2e-5,
        test_size=0.2
    )
