import os
import argparse
from pprint import pprint
from dotenv import load_dotenv
from langchain_teddynote import logging

from crew import Crew
from graph_relevancerag import RelevanceRAG
from graph_ensemblerag import EnsembleRAG
from graph_multiagentrag import MultiAgentRAG

from utils import load_config, save_output2json
from prompt import load_system_prompt, load_invoke_input

# .env 파일 로드
load_dotenv(dotenv_path=".env")

# API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# LangSmith 추적 기능을 활성화합니다. (선택적)
os.environ["LANGCHAIN_TRACING_V2"] = "true"


def get_rag_instance(
    rag_method, 
    file_folder, 
    file_number, 
    chunk_size, 
    chunk_overlap,
    search_k,
    system_prompt, 
    model_name, 
    save_graph_png
):
    """
    RAG 클래스를 동적으로 받아서 인스턴스를 생성하는 함수
    
    Params:
        rag_method: RAG 방법 ("relevance-rag", "ensemble-rag", "multiagent-rag")
        file_folder: 논문 파일이 위치한 폴더 경로
        file_number: 처리할 논문 번호
        system_prompt: system prompt
        model_name: LLM 모델 명 ("gpt-4o", "gpt-4o-mini")
        save_graph_png: graph 저장 결정
        
    Return:
        생성된 RAG 모델 인스턴스
    """
    
    # RAG 모델 인스턴스 생성
    if rag_method == "relevance-rag":
        return RelevanceRAG(file_folder, file_number, chunk_size, chunk_overlap, search_k, system_prompt, model_name, save_graph_png)
        
    elif rag_method == "ensemble-rag":
        return EnsembleRAG(file_folder, file_number, chunk_size, chunk_overlap, search_k, system_prompt, model_name, save_graph_png)
        
    elif rag_method == "multiagent-rag":
        return MultiAgentRAG(file_folder, file_number, chunk_size, chunk_overlap, search_k, system_prompt, model_name, save_graph_png)
    

def main(args):    
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
        crew = Crew(
            file_folder=f"{args.data_folder}/raw/", 
            file_number=file_number, 
            rag_method="crew-rag", 
            chunk_size=config["embedding_params"]["chunk_size"], 
            chunk_overlap=config["embedding_params"]["chunk_overlap"], 
            search_k=config["embedding_params"]["search_k"], 
            model_name=config["model_name"]        
        )
        sample_name_searcher_chain = crew.sample_name_searcher()
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
                print(result["discussion"])
                temp_answer = result["discussion"][0][category_names[category_number-1]]
            elif result.get("messages"):
                temp_answer = result["messages"][-1][category_names[category_number-1]]

            ## 결과 저장
            save_output2json(each_answer=temp_answer, file_num=file_number, rag_method=config["rag_method"], category_number=category_number)
                
            print(f"##          Print {file_number} Result:")
            print("------------------------------------------------------------------")
            pprint(temp_answer, sort_dicts=False)        
    
    
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
    args = parser.parse_args()

    main(args)
