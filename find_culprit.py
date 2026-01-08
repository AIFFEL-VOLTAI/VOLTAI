import os

def find_text_in_files(directory, text_to_find):
    print(f"📂 '{directory}' 폴더에서 '{text_to_find}' 검색 시작...\n")
    found = False
    
    # 검색할 파일 확장자 (필요하면 추가)
    target_extensions = ['.py', '.yaml', '.yml', '.json', '.env', '.txt']
    
    for root, dirs, files in os.walk(directory):
        # .git이나 가상환경 폴더 등은 제외 (속도 향상)
        if '.git' in root or 'venv' in root or 'miniconda' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if any(file.endswith(ext) for ext in target_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if text_to_find in line:
                                print(f"🚨 발견! 파일: {file_path}")
                                print(f"   └── {i+1}번째 줄: {line.strip()}")
                                found = True
                except Exception as e:
                    print(f"⚠️ 읽기 실패: {file_path} ({e})")

    if not found:
        print(f"\n✅ '{text_to_find}' 문자열을 찾을 수 없습니다. (메모리에 캐시된 문제일 수도 있습니다)")
    else:
        print(f"\n🔥 위의 파일들을 수정하여 '{text_to_find}'를 제거하세요!")

# 현재 폴더에서 검색 실행
if __name__ == "__main__":
    current_folder = os.getcwd()
    # "gpt-5.1"을 찾습니다.
    find_text_in_files(current_folder, "gpt-5.1")