from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid
import copy
import json
import yaml
from typing import Union

def load_invoke_input(config_folder:str="./config", category_number:int=1, rag_method:str="multiagent-rag", sample_names:list=None) -> Union[tuple, dict]:  
    """
    질문 파일을 불러오고, 주어진 RAG 방식에 따라 적절한 입력 형식을 반환합니다.

    Args:
        config_folder (str, optional): 설정 파일이 저장된 폴더 경로. 기본값은 "./config".
        category_number (int, optional): 불러올 질문 파일의 카테고리 번호. 기본값은 1.
        rag_method (str, optional): RAG 방식 (예: "multiagent-rag", "relevance-rag", "ensemble-rag"). 기본값은 "multiagent-rag".

    Raises:
        KeyError: 지원되지 않는 RAG 방식이 입력된 경우 예외 발생.

    Returns:
        Union[tuple, dict]: RAG 방식에 따라 적절히 구성된 입력 데이터.
    """     
    category_names = ["CAM (Cathode Active Material)", "Electrode (half-cell)", "Morphological Properties", "Cathode Performance"]

    question_file_name = f"c{category_number}-question.yaml"
    question_path = f"{config_folder}/{rag_method}/{question_file_name}"
    with open(question_path, 'r', encoding="utf-8") as file:
        question = yaml.safe_load(file)
    print(f"##          {question_path}를 불러왔습니다.")
        
    ## category 별 question 생성
    for i, sample_name in enumerate(sample_names):
        if category_number == 1:
            #question["template"][category_names[category_number-1]]["Stoichiometry information"][sample_name] = {}
            #question["template"][category_names[category_number-1]]["Commercial NCM used"][sample_name] = {}
            # [수정된 부분 시작] ------------------------------------------------
            target_category = category_names[category_number-1] # "CAM (Cathode Active Material)"
            
            # 첫 번째 샘플 처리 시: 원본 템플릿(스키마) 복사 후 부모 초기화
            if i == 0:
                base_schema = copy.deepcopy(question["template"][target_category])
                question["template"][target_category] = {}
            
            # 스키마를 샘플 이름 아래에 할당
            question["template"][target_category][sample_name] = copy.deepcopy(base_schema)
            # [수정된 부분 끝] --------------------------------------------------
        elif category_number == 2:
            #question["template"][category_names[category_number-1]][sample_name] = {}
            target_category = category_names[category_number-1] # "Electrode (half-cell)"
            
            # 첫 번째 샘플을 처리할 때만 스키마를 복사하고 틀을 비웁니다.
            if i == 0:
                # 1. yaml에 있던 원본 속성들(Active material 등)을 따로 저장
                base_schema = copy.deepcopy(question["template"][target_category])
                # 2. 원래 자리는 비워서 중복 출력 방지
                question["template"][target_category] = {}
            
            # 3. 저장해둔 스키마를 샘플 이름 아래에 넣어줌
            question["template"][target_category][sample_name] = copy.deepcopy(base_schema)
        elif category_number == 3:
            temp_template = question["template"][category_names[category_number-1]]
            for k in temp_template.keys():
                question["template"][category_names[category_number-1]][k][sample_name] = None
        elif category_number == 4:
            temp_performance = question["template"]["Cathode Performance"][""]
            question["template"]["Cathode Performance"].update({sample_name:temp_performance})
            if i == len(sample_names)-1:
                del question["template"]["Cathode Performance"][""]
    
    question_text = question['question_text']
    template = question['template'] 
    
# ---------------- [수정 핵심: Example 파일 미리 로드] ----------------
    # 모든 방식에서 Example을 사용하므로, 공통 로직으로 위로 뺐습니다.
    example_file_name = f"c{category_number}-example.json"
    example_path = f"{config_folder}/{rag_method}/{example_file_name}"
    
    try:
        with open(example_path, 'r', encoding="utf-8") as file:
            json_example = json.load(file)
    except FileNotFoundError:
        print(f"Warning: Example file not found at {example_path}")
        json_example = {} # 예외 처리
    # ------------------------------------------------------------------

    ## method 별 invoke_input 생성
    if rag_method == "multiagent-rag": 
        # [수정] 프롬프트에 Template과 Example을 모두 포함시킵니다.
        # json.dumps를 사용하여 dict를 깔끔한 문자열로 변환하는 것을 추천합니다.
        
        prompt_content = (
            f"{question_text}\n\n"
            f"Fill out in the `null`, `[]` and `{{}}` values based on the schema below:\n"
            f"{json.dumps([template], ensure_ascii=False, indent=2)}\n\n"
            f"Here is a reference EXAMPLE of the desired output format:\n"
            f"{json.dumps(json_example, ensure_ascii=False, indent=2)}"
        )

        invoke_input = (
            {"messages": [HumanMessage(
                content=prompt_content, 
                name="Researcher")]
            }, 
            {"recursion_limit": 100}
        )
                                            
    elif rag_method == "relevance-rag" or rag_method == "ensemble-rag":        
        # [수정] 위에서 이미 json_example을 로드했으므로 중복 로드 코드를 제거하고 변수만 사용합니다.
        
        config = RunnableConfig(
            recursion_limit=40, 
            configurable={"thread_id": random_uuid()}
            )

        invoke_input = {
            "input": {
                "question": f"{question_text}{[template]}".replace("'", '"'), 
                "example": AIMessage(content=[json_example])
            }, 
            "config": config
        }

    else: 
        raise KeyError(f"Unsupported rag_method: {rag_method}. Please use one of ['multiagent-rag', 'relevance-rag', 'ensemble-rag'].")
    
    return invoke_input