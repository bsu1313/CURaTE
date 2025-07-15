import json

# Path to your JSON file
json_file = "truthfulQA_continual_setting/TruthfulQA_split_ids.json"

# Load the JSON data
with open(json_file, "r") as f:
    data = json.load(f)

# Count entries for each stage
for stage, entries in data.items():
    print(f"{stage}: {len(entries)} entries")
# TruthfulQA: 272 + 272 + 273 = 817