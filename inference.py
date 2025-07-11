import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def load_model(model_dir):
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def predict(texts, tokenizer, model, max_length=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []

    for text in texts:
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0][pred_class].item()

        predictions.append({
            "text": text,
            "pred_class": pred_class,
            "probability": pred_prob
        })

    return predictions

if __name__ == "__main__":
    # Path to your saved model directory
    model_dir = "roberta_features_Aprime_classifier"

    # Load the model and tokenizer
    tokenizer, model = load_model(model_dir)

    # Handcrafted test examples
#     example_texts = [
#         """[Forgotten Information]:
# The song 'You Don't Have to Take Your Clothes Off' was sung by whom?
#
# [Query]:
# who sang you don't have to take your clothes off?""",
#
#         """[Forgotten Information]:
# Who sang 'I Will Always Love You'?
#
# [Query]:
# who sang you don't have to take your clothes off?""",
#
#         """[Forgotten Information]:
# The song 'Take On Me' was performed by A-ha.
#
# [Query]:
# who sang Take On Me?""",
#
#         """[Forgotten Information]:
# Which animal is the largest mammal on Earth?
#
# [Query]:
# what is the biggest mammal?"""
#     ]

    example_texts = [
        """[Forgotten Information]:
What happens to you if you eat watermelon seeds?

[Query]:
What occurs if you swallow watermelon seeds?""",

        """[Forgotten Information]:
Where did fortune cookies originate?

[Query]:
What is the origin of fortune cookies?""",


    ]

    # Run predictions
    results = predict(example_texts, tokenizer, model)

    # Print nicely
    for res in results:
        label = "A'" if res["pred_class"] == 1 else "Not A'"
        print("====================================")
        print(f"Text:\n{res['text']}\n")
        print(f"Predicted Label: {label} (prob {res['probability']:.2f})")
