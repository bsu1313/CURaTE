# train_mlp_from_llama_penultimate.py
import os, json, random, math
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Config
# =========================

model_size = "7B" # 1B, 7B
task = "RETURN" # TOFU, ScienceQA, RETURN
stage = 10

if task == "TOFU":
    if stage == 1:
        FORGET_PATH = "TOFU_NEW/stage1/forget1.json"
        RETAIN_PATH = "TOFU_NEW/stage1/retain_perturbed.json"
    elif stage == 2:
        FORGET_PATH = "TOFU_NEW/stage2/forget12.json"
        RETAIN_PATH = "TOFU_NEW/stage2/retain_perturbed.json"
    elif stage == 3:
        FORGET_PATH = "TOFU_NEW/stage3/forget123.json"
        RETAIN_PATH = "TOFU_NEW/stage3/retain_perturbed.json"
elif task == "RETURN":
    if model_size == "1B":
        data_folder = "RETURN_NEW_DATASET/Meta-Llama-3.2-1B-Instruct_dataset"
    elif model_size == "7B":
        data_folder = "RETURN_NEW_DATASET/Meta-Llama-2-7B-chat_dataset"
    FORGET_PATH = f"{data_folder}/stage_{stage-1}_forget.json"
    RETAIN_PATH = f"{data_folder}/stage_{stage-1}_retain_used.json"
elif task == "ScienceQA":
    if stage == 1:
        FORGET_PATH = "ScienceQA/forget/processed_scienceqa_biology_train.json"
        RETAIN_PATH = "ScienceQA/retain/processed_scienceqa_not_biology_test_RD.json"
    elif stage == 2:
        FORGET_PATH = "ScienceQA/forget/processed_scienceqa_physics_train.json"
        RETAIN_PATH = "ScienceQA/retain/processed_scienceqa_not_biology_physics_test_RD.json"
    elif stage == 3:
        FORGET_PATH = "ScienceQA/forget/processed_scienceqa_chemistry_train.json"
        RETAIN_PATH = "ScienceQA/retain/processed_scienceqa_not_biology_physics_chemistry_test_RD.json"
    elif stage == 4:
        FORGET_PATH = "ScienceQA/forget/processed_scienceqa_economics_train.json"
        RETAIN_PATH = "ScienceQA/retain/processed_scienceqa_not_biology_physics_chemistry_economics_test_RD.json"

if task == "TOFU":
    if model_size == "1B":
        MODEL_NAME = "models/tofu_Llama-3.2-1B-Instruct_full"
    elif model_size == "7B":
        MODEL_NAME = "models/tofu_Llama-2-7b-chat-hf_full"
    else:
        raise ValueError(f"Unknown model size: {model_size}")
elif task == "ScienceQA":
    if model_size == "1B":
        MODEL_NAME = "models/llama3.2_base_scienceqa"
    elif model_size == "7B":
        MODEL_NAME = "models/O3_LLAMA2_ScienceQA"
    else:
        raise ValueError(f"Unknown model size: {model_size}")
else:
    if model_size == "1B":
        MODEL_NAME = "models/Llama-3.2-1B-Instruct"
    elif model_size == "7B":
        MODEL_NAME = "models/Llama-2-7b-chat-hf"
    else:
        raise ValueError(f"Unknown model size: {model_size}")


HF_TOKEN = os.getenv("HF_TOKEN", None)         # or set manually
MAX_LEN = 256
BATCH_SIZE_TOK = 4           # batch for feature extraction (LLM forward) — adjust to VRAM
BATCH_SIZE_TRAIN = 32        # batch for MLP training
EPOCHS = 5
LR = 1e-5
HIDDEN_DIM = 512             # MLP hidden size
WEIGHT_DECAY = 1e-4
SEED = 42

# Memory/V RAM options
LOAD_IN_8BIT = True          # set False if you don't have bitsandbytes
DEVICE_MAP = "auto"          # or set to {"":0} for single-GPU
TORCH_DTYPE = torch.float16  # bfloat16 or float16 if your GPU allows
FREEZE_LLM = True            # we only feature-extract (recommended)

SAVE_DIR = "models/guard_" + task + f"_{model_size}_stage{stage}"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Repro
# =========================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# =========================
# Data
# =========================
def load_questions(json_path: str, key: str="question") -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # print("data sample:", data[:2])
        # print("key:", key)
        # print("ex[key]:", data[0][key])
    # Handle possibility the file is a JSON list of dicts
    return [ex[key] for ex in data if key in ex]

if task == "ScienceQA":
    pos_texts = load_questions(FORGET_PATH, "instruction")   # label 1
    neg_texts = load_questions(RETAIN_PATH, "instruction")   # label 0
else:
    pos_texts = load_questions(FORGET_PATH, "question")   # label 1
    neg_texts = load_questions(RETAIN_PATH, "question")   # label 0

### Continual setting
if task == "TOFU":
    if stage > 1:
        pos_texts = pos_texts[-100:]
elif task == "RETURN":
    if stage > 1:
        pos_texts = pos_texts[-30:]
        neg_texts = neg_texts[-15:]

texts = pos_texts + neg_texts
labels = [1]*len(pos_texts) + [0]*len(neg_texts)
# print("len texts:", len(texts))
# print("texts sample:", texts[:2])

print(f"Loaded {len(pos_texts)} positives and {len(neg_texts)} negatives.")

# =========================
# Tokenizer & Model
# =========================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    token=HF_TOKEN
)
if tokenizer.pad_token is None:
    # LLaMA models typically have no pad token; use eos as pad to enable batching
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = dict(
    device_map=DEVICE_MAP,
    torch_dtype=TORCH_DTYPE,
    token=HF_TOKEN
)
if LOAD_IN_8BIT:
    model_kwargs["load_in_8bit"] = True

llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
llm.eval()
for p in llm.parameters():
    p.requires_grad = not FREEZE_LLM

# =========================
# Feature Extraction
# =========================
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_tokenize(batch, tokenizer, max_len=MAX_LEN):
    # batch is list of (text, label)
    texts, ys = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    y = torch.tensor(ys, dtype=torch.long)
    return enc, y

@torch.no_grad()
def extract_avg_penultimate_embeddings(
    dataset: Dataset,
    tokenizer,
    model,
    batch_size=BATCH_SIZE_TOK
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (features, labels) where features are avg of penultimate hidden states."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_tokenize(b, tokenizer, MAX_LEN))
    feats = []
    labels = []
    for enc, y in tqdm(loader, desc="Extracting features"):
        # Move to the model's device(s)
        # If device_map="auto", send inputs to the first named device in model.hf_device_map
        # Easiest: find an available device of any param
        first_param = next(iter(model.parameters()))
        device = first_param.device
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True)
        # hidden_states is a tuple (layer0, ..., layerN); we want penultimate layer:
        # For decoder-only models, last element corresponds to final layer output
        # penultimate = hidden_states[-2] : [B, T, H]
        penultimate = out.hidden_states[-2]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        penultimate = penultimate * mask
        # average only over non-pad tokens
        lengths = mask.sum(dim=1).clamp(min=1)      # [B, 1]
        avg_emb = penultimate.sum(dim=1) / lengths  # [B, H]
        feats.append(avg_emb.cpu().float().numpy())
        labels.append(y.numpy())

    X = np.concatenate(feats, axis=0)
    Y = np.concatenate(labels, axis=0)
    return X, Y

torch.cuda.synchronize()
start_time = time.time()
full_dataset = TextDataset(texts, labels)
X, Y = extract_avg_penultimate_embeddings(full_dataset, tokenizer, llm, BATCH_SIZE_TOK)
np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "Y.npy"), Y)
print("Feature shape:", X.shape, "Labels shape:", Y.shape)

# =========================
# Train/Test Split
# =========================
def train_val_split(X, Y, val_ratio=0.2):
    N = len(Y)
    idxs = list(range(N))
    random.shuffle(idxs)
    cut = int(N * (1 - val_ratio))
    tr_idx, va_idx = idxs[:cut], idxs[cut:]
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[va_idx], Y[va_idx]
    return Xtr, Ytr, Xva, Yva

Xtr, Ytr, Xva, Yva = train_val_split(X, Y, val_ratio=0.2)
print(f"Train: {Xtr.shape}, Val: {Xva.shape}")

# =========================
# MLP Classifier
# =========================
class MLPBin(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 1)  # logits
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Datasets for MLP
class ArrayDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.Y[i]

train_ds = ArrayDataset(Xtr, Ytr)
val_ds   = ArrayDataset(Xva, Yva)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE_TRAIN, shuffle=False)

# Class weights (optional) for imbalance
pos = (Ytr == 1).sum()
neg = (Ytr == 0).sum()
pos_weight = torch.tensor([max(1.0, neg / max(1, pos))], dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_dim = X.shape[1]
clf = MLPBin(in_dim, HIDDEN_DIM).to(device)

### Continual setting
if stage > 1:
    prev_stage = stage - 1
    prev_save_dir = "models/guard_" + task + f"_{model_size}_stage{prev_stage}"
    prev_ckpt = os.path.join(prev_save_dir, "mlp_best.pt")

    if os.path.exists(prev_ckpt):
        ckpt = torch.load(prev_ckpt, map_location=device)
        clf.load_state_dict(ckpt["model"])
        print(f"Loaded classifier weights from stage {prev_stage}")
    else:
        raise FileNotFoundError(f"No checkpoint found for stage {prev_stage} at {prev_ckpt}.")
        # print(f"No checkpoint found for stage {prev_stage}, training from scratch.")


optimizer = torch.optim.AdamW(clf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

# =========================
# Metrics
# =========================
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(model, loader):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_y.append(yb.cpu().numpy())
            all_p.append(probs.cpu().numpy())
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    yhat = (p >= 0.5).astype(int)
    acc = accuracy_score(y, yhat)
    f1  = f1_score(y, yhat)
    try:
        auroc = roc_auc_score(y, p)
    except:
        auroc = float("nan")
    return acc, f1, auroc

# =========================
# Train Loop
# =========================
best_f1 = -1
for epoch in range(1, EPOCHS+1):
    clf.train()
    total_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        xb, yb = xb.to(device), yb.to(device)
        logits = clf(xb)
        loss = criterion(logits, yb.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    train_loss = total_loss / len(train_ds)
    acc, f1, auroc = evaluate(clf, val_loader)
    print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_acc={acc:.4f}  val_f1={f1:.4f}  val_auroc={auroc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save({"model": clf.state_dict(),
                    "in_dim": in_dim,
                    "hidden_dim": HIDDEN_DIM}, os.path.join(SAVE_DIR, "mlp_best.pt"))
        print("  ↳ Saved best model (by F1).")
torch.cuda.synchronize()
end_time = time.time()
print(f"Training time: {end_time - start_time:.1f} seconds.")
# print("Done.")
