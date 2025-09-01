import json

# Input and output file names
# input_file = "NQ_CURE_18K_a.json"
# output_file = "NQ_CURE_18K_a_no_b.json"
input_file = "NQ_CURE_NO_HN_18K_a.json"
output_file = "NQ_CURE_NO_HN_18K_a_no_b.json"


def remove_b_prime_records(input_file, output_file):
    # Load the JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter out records where "features" == "B'"
    filtered_data = [record for record in data if record.get("features") != "B'"]

    # Save the filtered data to a new file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"Saved filtered data to {output_file} (removed {len(data) - len(filtered_data)} records).")

if __name__ == "__main__":
    remove_b_prime_records(input_file, output_file)
