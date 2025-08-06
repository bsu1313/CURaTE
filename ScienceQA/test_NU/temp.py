import os
import json

# JSON 파일들이 저장된 폴더 경로 설정
folder_path = '.'  # 여기에 실제 폴더 경로 입력

# 알파벳 정답들을 저장할 리스트
contrastive_answers = []

# 폴더 내 모든 JSON 파일에 대해 작업
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        modified = False

        # 데이터가 리스트일 경우만 처리
        if isinstance(data, list):
            for item in data:
                answer_text = item.get("contrastive_answer", "")
                if "The answer is " in answer_text:
                    # 알파벳만 추출해서 덮어쓰기
                    answer_letter = answer_text.strip().split()[-1].strip(".")
                    item["contrastive_answer"] = answer_letter
                    modified = True

        # 수정한 경우 저장
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

print("모든 JSON 파일의 contrastive_answer 값을 정답 알파벳으로 덮어썼습니다.")
