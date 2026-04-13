import json
import uuid

# Load your JSON data
with open("NQ_CURaTE_21k.json", "r") as f:
    data = json.load(f)

new_entries = []
for entry in data:
    # Get the original query from the first human conversation turn
    conv = entry["conversations"][0]["value"]
    # Extract the [Query] text
    query_start = conv.find("[Query]:")
    if query_start != -1:
        query_text = conv[query_start + len("[Query]:"):].strip()

        # Create a new entry with Forgotten Info == Query
        new_entry = {
            "id": str(uuid.uuid4()),
            "group_id": entry["group_id"],
            "features": "A'",
            "conversations": [
                {
                    "from": "human",
                    "value": f"Based on the [Forgotten Information], decide whether to answer or refuse to answer the [Query]. Then provide an appropriate response accordingly.\n\n[Forgotten Information]:\n{query_text}\n\n[Query]:\n{query_text}"
                },
                {
                    "from": "gpt",
                    "value": "I don't have information about that."
                }
            ]
        }
        new_entries.append(new_entry)

# Append the new examples to your data
data.extend(new_entries)

# Save the augmented dataset
with open("NQ_CURaTE_21k_augmented.json", "w") as f:
    json.dump(data, f, indent=2)