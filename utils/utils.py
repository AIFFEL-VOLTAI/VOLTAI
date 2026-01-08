import os
import yaml
import json
import time
import pandas as pd
from typing import Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid

from langsmith import Client

class LangSmithLogger:
    def __init__(self, project_name: str | None = None):
        self.project_name = project_name or "voltai-rag"  # 프로젝트 이름은 마음대로
        # Client는 환경변수(LANGCHAIN_API_KEY 등)를 자동으로 읽음
        self.client = Client()

    def langsmith(self, message: str):
        """LangSmith에 단순 로그(run) 하나 남기는 메서드"""
        run = self.client.create_run(
            project_name=self.project_name,
            name="main-run",
            inputs={"message": message},
            run_type="chain",
        )
        print(f"[LangSmith] {message}")
        return run

# main.py에서 from utils.utils import logging 으로 쓰는 객체
logging = LangSmithLogger()



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

import os
import time
import yaml
import json
from typing import Union
from pprint import pprint

import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage
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




def save_data2output_folder(output_folder: str, data, filename: str, hyper_param_method: str):
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
        if hyper_param_method:
            filename += f"-{hyper_param_method}.json"        
        else: 
            filename += f"-{timestamp}.json"        

        file_path = os.path.join(output_folder, filename)
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"##       {file_path}를 저장했습니다.")
        
    else:
        print("    데이터 형식이 지원되지 않습니다. pandas DataFrame 또는 dict만 저장 가능합니다.")
        



def save_output2json(each_answer: dict, file_num: int, rag_method: str, category_number: int, hyper_param_method: str):    
    ## 파일 이름 설정
    json_file_num = f"00{file_num}"[-3:]
        
    json_name = f"paper_{json_file_num}_output"

    save_data2output_folder(output_folder=f"./output/json/{rag_method}/{json_name}/", data=each_answer, filename=f"category-{category_number}-{json_name}", hyper_param_method=hyper_param_method)


