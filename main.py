import os
import argparse
from dotenv import load_dotenv
from langchain_teddynote import logging

from models.crew import Crew
from models.instance import get_rag_instance

from prompt.system_prompt import load_system_prompt
from prompt.invoke import load_invoke_input

from utils.utils import *

# 환경 변수 로드
load_dotenv(dotenv_path=".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# LangSmith 추적 기능 활성화
os.environ["LANGCHAIN_TRACING_V2"] = "true"


def main(args):
    category_names = ["CAM (Cathode Active Material)", "Electrode (half-cell)", "Morphological Properties", "Cathode Performance"]
    config = load_config(args.config_folder)
    logging.langsmith(f"{config['project_name']} : {config['rag_method']}")
    
    file_folder = f"{args.data_folder}/input_data"
    file_num_list = config["file_num_list"] if not config["process_all_files"] else [int(f[6:9]) for f in os.listdir(file_folder) if os.path.isfile(os.path.join(file_folder, f))]
    
    for file_number in file_num_list:
        print(f"##### {file_number}번째 논문 #####")
        crew = Crew(file_folder, file_number, config["rag_method"], **config["embedding_params"], model_name=config["model_name"])
        sample_names = crew.sample_name_searcher().invoke(config["sample_name_searcher_question"])
        print(f"##       Sample Names    : {sample_names}")
        for category_number in range(1, 5):
            category_name = category_names[category_number-1]
            print(f"## Category: {category_name}")
            system_prompt = load_system_prompt(config_folder=args.config_folder, category_number=category_number, rag_method=config["rag_method"])
            invoke_input = load_invoke_input(config_folder=args.config_folder, category_number=category_number, rag_method=config["rag_method"], sample_names=sample_names)
            
            rag = get_rag_instance(config["rag_method"], file_folder, file_number, **config["embedding_params"], system_prompt=system_prompt, model_name=config["model_name"], save_graph_png=config["save_graph_png"]).graph

            ## 질문이 딕셔너리 형태일 경우와 아닌 경우를 처리
            if isinstance(invoke_input, dict):
                result = rag.invoke(**invoke_input)
            else:
                result = rag.invoke(*invoke_input)

            ## RAG method에 따른 결과 확인
            if result.get("answer"):
                temp_answer = result["answer"][0][category_names[category_number-1]]
            elif result.get("discussion"):
                print(result["discussion"])
                temp_answer = result["discussion"][0][category_names[category_number-1]]
            elif result.get("messages"):
                temp_answer = result["messages"][-1][category_names[category_number-1]]
            save_output2json(temp_answer, file_number, config["rag_method"], category_number)
            print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config_folder", default="./config", type=str, help="config folder path")
    parser.add_argument("-df", "--data_folder", default="./data", type=str, help="data folder path")
    args = parser.parse_args()
    main(args)
