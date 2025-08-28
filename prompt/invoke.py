from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid

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
            question["template"][category_names[category_number-1]]["Stoichiometry information"][sample_name] = {}
            question["template"][category_names[category_number-1]]["Commercial NCM used"][sample_name] = {}
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
    
    ## method 별 invoke_input 생성
    if rag_method == "multiagent-rag": 
        invoke_input = (
            {"messages": [HumanMessage(
                content=f"{question_text}{[template]}".replace("'", '"'), 
                name="Researcher")]
            }, 
            {"recursion_limit": 30}
        )
                                        
    elif rag_method == "relevance-rag" or rag_method == "ensemble-rag":        
        example_file_name = f"c{category_number}-example.json"
        example_path = f"{config_folder}/{rag_method}/{example_file_name}"

        with open(example_path, 'r', encoding="utf-8") as file:
            json_example = json.load(file)
        

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