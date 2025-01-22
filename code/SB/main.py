import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid

import argparse

from graph import DataExtractor
from utils import load_question, outputs2csv, outputs2pprint, outputs2json


# .env 파일 로드
load_dotenv(dotenv_path=".env")

# API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# LangSmith 추적 기능을 활성화합니다. (선택적)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"


def main(args):
    ## question 불러오기
    question_list = load_question()
    
    total_outputs = {}
    if args.all_files: ## 모든 논문에서 추출할 때
        file_folder = args.file_folder
        file_num_list = [int(f[6:9]) for f in os.listdir(file_folder) if os.path.isfile(os.path.join(file_folder, f))]
        
    else: ## 지정된 논문에서 추출할 때
        file_num_list = args.file_list
        
    for file_num in file_num_list:
        print(f"#####   {file_num}번째 논문    #####")
        
        # graph 불러오기
        voltai_graph = DataExtractor(file_folder=args.file_folder, file_number=file_num).graph

        # config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(
            recursion_limit=args.recursion_limit, 
            # configurable={"thread_id": str(uuid.uuid4())}
            configurable={"thread_id": random_uuid()}
            )
        
        # 4개의 질문에 대해 그래프 실행 및 출력
        results = []
        for i, question in enumerate(question_list):
            print(f"    {i+1}번째 질문")
            result = voltai_graph.invoke(input={"question":question}, config=config)
            results.append(result)
        
        total_outputs[file_num] = results        

        outputs2pprint(total_outputs)      
          
        if args.mode == "json":
            outputs2json(total_outputs, file_num=file_num)
    
    if args.mode == "csv":
        outputs2csv(total_outputs=total_outputs, filename=args.file_name)            
        
    
if __name__ == "__main__":   
    def str2bool(value):
        if value.lower() in ('true', '1', 't', 'y', 'yes'):
            return True
        elif value.lower() in ('false', '0', 'f', 'n', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean values expected')
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-ff", "--file_folder", default="./data/input_data/", type=str, help="input data folder path")
    parser.add_argument("-fl", "--file_list", default=[56, 139], nargs='+', type=int, help="input data number")
    parser.add_argument("-rl", "--recursion_limit", default=20, type=int, help="maximum cycle of relevance check recursion")
    parser.add_argument("-fn", "--csv_file_name", default="temp_name", type=str, help="name of csv file")
    parser.add_argument("-m", "--mode", default="json", choices=["csv", "json"], type=str, help="choose output form")
    parser.add_argument("-a", "--all_files", default=False, type=str2bool, nargs='?', help="process to all papers")
    args = parser.parse_args()

    main(args)