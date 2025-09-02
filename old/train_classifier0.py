import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import random
import os
import sys

# # Define the binary classifier model
# class BinaryClassifier(nn.Module):
#     def __init__(self, input_dim):
#         super(BinaryClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 1)  # Output 1 value for binary classification
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return self.sigmoid(x)

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # Additional activation function
        x = self.fc3(x)
        return self.sigmoid(x)

# 1. Load JSON and split
def load_and_split_data(json_path, test_size=0.2, random_state=42, cache_path="cached_embeddings.json"):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert embeddings and labels back to appropriate types
        for item in data:
            item["f_emb"] = torch.tensor(item["f_emb"], dtype=torch.float32).tolist()
            item["q_emb"] = torch.tensor(item["q_emb"], dtype=torch.float32).tolist()
    else:
        data = []
        # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        model = SentenceTransformer('sentence-transformers-testing/stsb-bert-tiny-safetensors')
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for item in raw:
                for turn in item["conversations"]:
                    if turn["from"] == "human":
                        full_text = turn["value"]
                        if "\n\n" in full_text:
                            human_text = full_text.split("\n\n", 1)[1].strip()
                        else:
                            human_text = full_text.strip()
                        f_info = human_text.split("\n\n", 1)[0].strip()
                        query = human_text.split("\n\n", 1)[1].strip()
                        f_info = f_info.split("\n", 1)[1].strip()
                        query = query.split("\n", 1)[1].strip()
                        f_emb = model.encode(f_info, convert_to_tensor=True)
                        q_emb = model.encode(query, convert_to_tensor=True)

                label = 1 if item["features"] == "A'" else 0

                data.append({
                    "f_info": f_info,
                    "query": query,
                    "f_emb": f_emb.tolist(),
                    "q_emb": q_emb.tolist(),
                    "label": label
                })
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    return train_data, test_data


# Define a custom dataset
class EmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        f_emb = torch.tensor(item['f_emb'], dtype=torch.float32)
        q_emb = torch.tensor(item['q_emb'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.float32)
        # Concatenate the embeddings as input
        return torch.cat((f_emb, q_emb), dim=0), label


# Load and split data
# train_data, test_data = load_and_split_data("NQ_LTU_18k_augmented.json")
# train_data, test_data = load_and_split_data("NQ_LTU_18k.json")
train_data, test_data = load_and_split_data("../NQ_LTU_21k_augmented.json")

# Create DataLoader
train_dataset = EmbeddingDataset(train_data)
test_dataset = EmbeddingDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize the model, loss function, and optimizer
input_dim = len(train_data[0]['f_emb']) + len(train_data[0]['q_emb'])  # Dimension of concatenated embeddings
model = BinaryClassifier(input_dim)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # 0.66
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # 0.87
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 0.92 0.91 0.91
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) # 0.83 0.88

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()  # Apply threshold of 0.5
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# Create the folder if it doesn't exist
os.makedirs('binary_classifier', exist_ok=True)

# Save input_dim for future use
with open("binary_classifier/config.json", "w") as f:
    json.dump({"input_dim": input_dim}, f)

# Save the model state_dict inside the 'binary_classifier' folder
torch.save(model.state_dict(), 'binary_classifier/binary_classifier.pth')