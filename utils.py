import os
import time
import yaml
import json
from typing import Union
from pprint import pprint

import pandas as pd

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid


def load_config(config_folder:str="./config") -> dict:
    """
    config 파일을 불러오는 함수입니다. 

    Args:
        config_folder (str, optional): 설정 파일이 저장된 폴더 경로. Defaults to "./config".

    Returns:
        dict: YAML 파일 내용을 Python 딕셔너리로 변환하여 반환.
    """
    file_path = f"{config_folder}/config.yaml"
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    print(f"## {file_path}를 불러왔습니다.")
    
    return config


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
    
    print(f"##          {system_prompt_path}를 불러왔습니다.")
    
    return system_prompt


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
    
    example_file_name = f"c{category_number}-example.json"
    example_path = f"{config_folder}/{rag_method}/{example_file_name}"
    with open(example_path, 'r', encoding="utf-8") as file:
        json_example = json.load(file)
    print(f"##          {example_path}를 불러왔습니다.")
    
    if rag_method == "multiagent-rag": 
        invoke_input = (
            {"messages": [HumanMessage(content=question["question"], name="Researcher")]}, 
            {"recursion_limit": 30}
        )
    
    elif rag_method == "relevance-rag" or rag_method == "ensemble-rag":
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
            
        config = RunnableConfig(
            recursion_limit=30, 
            configurable={"thread_id": random_uuid()}
            )

        invoke_input = {
            "input": {
                "question": f"{question['question_text']}{[question['template']]}".replace("'", '"'), 
                "example": json_example
            }, 
            "config": config
        }

    else: 
        raise KeyError(f"Unsupported rag_method: {rag_method}. Please use one of ['multiagent-rag', 'relevance-rag', 'ensemble-rag'].")
    
    return invoke_input


def save_data2output_folder(output_folder: str, data, filename: str):
    """
    Save data as either a CSV or JSON file in a specified output folder with a timestamped filename.

    Args:
        output_folder (str): The path to the output folder.
        data: The data to save (pandas DataFrame for CSV, dict for JSON).
        filename (str): The base name of the file (without timestamp and extension).
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ## 파일 이름 중 시간 설정
    timestamp = time.strftime("%y%m%d%H%M%S")
    
    ## 파일 유형에 따라 결정
    if isinstance(data, pd.DataFrame):
        file_path = os.path.join(output_folder, f"{timestamp}-{filename}.csv")
        data.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"    CSV 파일 {file_path}에 저장되었습니다.")
        
    elif isinstance(data, dict):
        file_path = os.path.join(output_folder, f"{filename}-{timestamp}.json")
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"##       {file_path}를 저장했습니다.")
        
    else:
        print("    데이터 형식이 지원되지 않습니다. pandas DataFrame 또는 dict만 저장 가능합니다.")


def save_output2json(each_answer:dict, file_num:int, rag_method:str, category_number:int):    
    ## 파일 이름 설정
    json_file_num = f"000000{file_num}"[-3:]
    # if file_num < 10:
    #     json_file_num = f"00{file_num}"
    # elif file_num < 100:
    #     json_file_num = f"0{file_num}"
    # else:
    #     json_file_num = f"{file_num}"
        
    json_name = f"paper_{json_file_num}_output"
    
    save_data2output_folder(output_folder=f"./output/json/{rag_method}/{json_name}/", data=each_answer, filename=f"category-{category_number}-{json_name}")


# def outputs2csv(total_outputs:dict, filename="temp_result") -> pd.DataFrame:
#     answers_list = []
#     for file_num in list(total_outputs.keys()):
#         outputs = total_outputs[file_num]
#         answers = outputs[0]["answer"][0]["CAM (Cathode Active Material)"] | outputs[1]["answer"][0]["Electrode (only for coin-cell (half-cell))"] | outputs[2]["answer"][0]["Morphological results"] | outputs[3]["answer"][0]["Cathode Performance"]
#         answers["Paper Number"] = file_num
#         answers_list.append(answers)

#     answers_csv = pd.DataFrame(answers_list)                               
#     columns = ["Paper Number"] + [col for col in answers_csv.columns if col != "Paper Number"]
#     answers_csv = answers_csv[columns]
    
#     save_data2output_folder(output_folder="./output/csv/", data=answers_csv, filename=filename)
    
    
# def outputs2pprint(total_outputs:dict):
#     for file_num in list(total_outputs.keys()):
#         print(f"    {file_num}번째 논문 결과")
#         outputs = total_outputs[file_num]
#         answers = outputs[0]["answer"][0] | outputs[1]["answer"][0] | outputs[2]["answer"][0] | outputs[3]["answer"][0]
#         pprint.pprint(answers, sort_dicts=False)