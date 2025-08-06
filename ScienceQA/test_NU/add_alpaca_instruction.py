import os
import json

def apply_template_to_all_json_files_in_folder(folder_path='.'):
    # 템플릿 정의
    template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )

    # 폴더 내 모든 파일 확인
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 리스트 형식인지 확인
                if isinstance(data, list):
                    for item in data:
                        instruction_text = item.get('contrastive_instruction', '')
                        item['contrastive_instruction_input'] = template.format(instruction=instruction_text)

                    # 동일 파일에 덮어쓰기
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"Updated: {filename}")
                else:
                    print(f"Skipped (not a list): {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 현재 폴더 기준 실행
apply_template_to_all_json_files_in_folder()
