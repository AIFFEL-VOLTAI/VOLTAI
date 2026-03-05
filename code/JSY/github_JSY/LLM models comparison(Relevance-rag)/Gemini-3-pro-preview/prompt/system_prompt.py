import yaml

def load_system_prompt(config_folder:str="./config", category_number:int=1, rag_method:str="multiagent-rag") -> dict:
    """
    시스템 프롬프트 구성 파일을 주어진 RAG 방식(rag_method)과 카테고리 번호(category_number)에 따라 불러옵니다.

    Args:
    config_folder (str, optional): 설정 파일이 저장된 폴더 경로. 기본값은 "./config".
    category_number (int, optional): 불러올 시스템 프롬프트의 카테고리 번호. 기본값은 1.
    rag_method (str, optional): RAG 방식 (예: "multiagent-rag", "relevance-rag"). 기본값은 "multiagent-rag".

    Returns:
    dict: YAML 파일 내용을 Python 딕셔너리로 변환하여 반환.
    """      
    
    file_name = f"c{category_number}-system-prompt.yaml"
    system_prompt_path = f"{config_folder}/{rag_method}/{file_name}"
    
    with open(system_prompt_path, 'r', encoding="utf-8") as file:
        system_prompt = yaml.safe_load(file)
    
    
    
    return system_prompt