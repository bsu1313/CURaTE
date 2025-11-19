import json
import csv

# Paths to your files
json_path = "truthfuQA_consent_false_only_augmented_llama_gen_consent_true_only.json"
csv_path = "TruthfulQA.csv"
output_path = "truthfulQA_enriched.json"

# 1. Load the JSON data
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 2. Load the CSV data into a dictionary keyed by Question
csv_data = {}
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = row["Question"].strip()
        csv_data[question] = {
            # "Best Answer": row["Best Answer"],
            "Correct Answers": row["Correct Answers"],
            "Incorrect Answers": row["Incorrect Answers"]
        }

# 3. Go through each item in the JSON and enrich it
enriched_data = []
for item in json_data:
    question = item["question"].strip()

    if question in csv_data:
        # Add the new fields
        item.update(csv_data[question])
    else:
        # If no match, you can decide what to do: skip, warn, or add empty fields
        print(f"⚠️ Warning: Question not found in CSV: {question}")
        # item["Best Answer"] = ""
        item["Correct Answers"] = ""
        item["Incorrect Answers"] = ""

    enriched_data.append(item)

# 4. Save the enriched JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(enriched_data, f, ensure_ascii=False, indent=2)

print(f"✅ Enriched data saved to {output_path}")
