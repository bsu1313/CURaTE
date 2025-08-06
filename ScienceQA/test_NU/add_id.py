import json

# 파일 경로
input_file = 'NU_scienceqa_biology_train_SD.json'
output_file = 'NU_scienceqa_biology_train_SD.json'

# JSON 파일 읽기
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 각 항목의 id에 10000 더하기
for item in data:
    if 'id' in item:
        item['id'] -= 10000
        item['id'] += 1000000

# 결과를 새로운 JSON 파일로 저장
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Updated JSON saved to {output_file}")
