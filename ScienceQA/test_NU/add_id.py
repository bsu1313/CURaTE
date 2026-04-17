import json


input_file = 'NU_scienceqa_biology_train_SD.json'
output_file = 'NU_scienceqa_biology_train_SD.json'


with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)


for item in data:
    if 'id' in item:
        item['id'] -= 10000
        item['id'] += 1000000

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Updated JSON saved to {output_file}")
