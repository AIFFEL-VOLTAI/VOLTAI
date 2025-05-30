import os
import argparse
import time
from pprint import pprint
from dotenv import load_dotenv
from langchain_teddynote import logging

from models import sample_name_searcher, get_rag_instance

from utils import load_config, save_output2json
from prompt import load_system_prompt, load_invoke_input

# .env 파일 로드
load_dotenv(dotenv_path=".env")

# API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# LangSmith 추적 기능을 활성화합니다. (선택적)
os.environ["LANGCHAIN_TRACING_V2"] = "true"    


def main(args):    
    ## 시간 측정 시작
    start_time = time.time()

    category_names = ["CAM (Cathode Active Material)", "Electrode (half-cell)", "Morphological Properties", "Cathode Performance"]

    ## config 불러오기
    config = load_config(config_folder=args.config_folder)
    
    ## 프로젝트 이름 설정: langsmith 추척
    logging.langsmith(f"{config['project_name']} : {config['rag_method']}")
    
    ## 전체 파일에 대해 진행여부 결정
    if config["process_all_files"]:
        file_folder = f"{args.data_folder}/raw"
        file_num_list = [int(f[6:9]) for f in os.listdir(file_folder) if os.path.isfile(os.path.join(file_folder, f))]
    else:
        file_num_list = config["file_num_list"]

    ## 각 논문에 대해 반복
    for file_number in file_num_list:
        print(f"#####    {file_number}번째 논문    #####")
        print(f"##       rag method     : {config['rag_method']}")

        ## Sample Name Searcher
        sample_name_searcher_chain = sample_name_searcher(
            file_folder=f"{args.data_folder}/raw/", 
            file_number=file_number, 
            chunk_size=config["embedding_params"]["chunk_size"], 
            chunk_overlap=config["embedding_params"]["chunk_overlap"], 
            search_k=config["embedding_params"]["search_k"], 
            model_name=config["model_name"]        
        )
        sample_names = sample_name_searcher_chain.invoke(config["sample_name_searcher_question"])
        print(f"##       Sample Names    : {sample_names}")
    
        for category_number in range(1,5):
            print(f"##          Category Name   : {category_names[category_number-1]}")

            ## config 파일과 system_prompt 와 invoke_input 불러오기
            system_prompt = load_system_prompt(config_folder=args.config_folder, category_number=category_number, rag_method=config["rag_method"])
            invoke_input = load_invoke_input(config_folder=args.config_folder, category_number=category_number, rag_method=config["rag_method"], sample_names=sample_names)
    
            ## graph 호출
            voltai_graph = get_rag_instance(
                rag_method=config["rag_method"], 
                file_folder=f"{args.data_folder}/raw/", 
                file_number=file_number, 
                chunk_size=config["embedding_params"]["chunk_size"], 
                chunk_overlap=config["embedding_params"]["chunk_overlap"], 
                search_k=config["embedding_params"]["search_k"], 
                system_prompt=system_prompt,
                model_name=config["model_name"], 
                discussion_model_name=config["discussion_model_name"],
                save_graph_png=config["save_graph_png"],
            ).graph
            
            ## 질문이 딕셔너리 형태일 경우와 아닌 경우를 처리
            if isinstance(invoke_input, dict):
                result = voltai_graph.invoke(**invoke_input)
            else:
                result = voltai_graph.invoke(*invoke_input)

            ## RAG method에 따른 결과 확인
            if result.get("answer"):
                temp_answer = result["answer"][0][category_names[category_number-1]]
            elif result.get("discussion"):
                temp_answer = result["discussion"][0][category_names[category_number-1]]
            elif result.get("messages"):
                temp_answer = result["messages"][-1][category_names[category_number-1]]

            ## 결과 저장
            save_output2json(each_answer=temp_answer, file_num=file_number, rag_method=config["rag_method"], category_number=category_number, hyper_param_method=args.hyper_param_method)
                
            pprint(temp_answer, sort_dicts=False)        

    ## 시간 측정 끝
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"코드 실행 시간: {execution_time:.4f} 초")

if __name__ == "__main__":   
    def str2bool(value):
        if value.lower() in ('true', '1', 't', 'y', 'yes'):
            return True
        elif value.lower() in ('false', '0', 'f', 'n', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean values expected')
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config_folder", default="./config", type=str, help="config folder path")     
    parser.add_argument("-df", "--data_folder", default="./data", type=str, help="data folder path")
    parser.add_argument("-hp", "--hyper_param_method", default=None, type=str, help="method of hyper parameters")
    args = parser.parse_args()

    main(args)
