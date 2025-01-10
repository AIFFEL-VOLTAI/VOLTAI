import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid
import pprint
import argparse

from graph import DataExtractor
from utils import outputs2csv, outputs2pprint


# .env 파일 로드
load_dotenv(dotenv_path=".env")

# API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# LangSmith 추적 기능을 활성화합니다. (선택적)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"


# 4개의 질문 입력
questions = [
    """Below are instructions for filling out items by referring to the examples.
    The values shown to the right of the colon (“:”) are examples;
    please delete them after reviewing and rewrite them with the values found in the PDF.
    If any item is not mentioned in the PDF, do not remove it—write “None.”
    [
        {
            "CAM (Cathode Active Material)": {
                "Stoichiometry information": "NCM-622",
                "Commercial NCM": "No",
                "Lithium source": "LiOH",
                "Synthesis method": "co-precipitation",
                "Crystallization method": "Hydrothermal",
                "Crystallization temperature": "100°C",
                "Crystallization time": "12 hr",
                "Doping": "Zr4+ doping",
                "Coating": "ZrO2 coating",
                "Additional treatment": "None"
                }
        }
    ]
    """,    
    
    """Below are instructions for filling out items by referring to the examples.
    The values shown to the right of the colon (“:”) are examples;
    please delete them after reviewing and rewrite them with the values found in the PDF.
    If any item is not mentioned in the PDF, do not remove it—write “None.”
    [
        {
            "Electrode (only for coin-cell (half-cell))": {
                "Active material : Conductive additive : Binder ratio": "90 : 5 : 5",
                "Electrolyte": "LiPF6 (EC, EMC, DEC mixture in a 1:1:1 volume ratio)",
                "Additive": "FEC 10% addition",
                "Electrode thickness": "100 µm",
                "Only Cathode Electrode diameter": "14π",
                "Loading density (mass loading of NCM)": "0.005 g/cm^2",
                "Additional treatment for electrode": "None"
                },
        }
    ]
    """,

    """Below are instructions for filling out items by referring to the examples.
    The values shown to the right of the colon (“:”) are examples;
    please delete them after reviewing and rewrite them with the values found in the PDF.
    If any item is not mentioned in the PDF, do not remove it—write “None.”
    [
        {        
            "Morphological results": {
                "Explanation of SEM results": "Fig. 2a, b; the NCM-622 seems to have more or less a spherical morphology with a diameter of 3–5 µm, composed of densely packed primary particles",
                "Explanation of TEM results": "None"
                },    
        }
    ]
    """,

    """Below are instructions for filling out items by referring to the examples.
    The values shown to the right of the colon (“:”) are examples;
    please delete them after reviewing and rewrite them with the values found in the PDF.
    If any item is not mentioned in the PDF, do not remove it—write “None.”
    [
        {            
            "Cathode Performance": {
                "Capacity at all C-rate, mAh/g (with electrode state)": [{
                "214.5 mAh/g": "@0.1C, ZrO2-coated",
                "200.8 mAh/g": "@0.5C, ZrO2-coated"
                }],
                "Voltage range": "2.8–4.3 V",
                "Temperature": "Room temperature and 55°C"
                }
        }
    ]
    """
]


def main(args):
    total_outputs = {}
    if args.all_files:
        folder_path = args.folder_path
        file_num_list = [int(f[6:9]) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
    else: 
        file_num_list = args.file_list
        
    for file_num in file_num_list:
        print(f"#####   {file_num}번째 논문    #####")
        
        # graph 불러오기
        voltai_graph = DataExtractor(
            folder_path=args.folder_path, 
            file_number=file_num
        ).graph

        # config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(
            recursion_limit=args.recursion_limit, 
            configurable={"thread_id": random_uuid()}
        )
        
        # 4개의 질문에 대해 그래프 실행 및 출력
        results = []
        for i, question in enumerate(questions):
            # inputs = GraphState(question=question)
            print(f"{i+1}번째 질문")
            # print(question)
            result = voltai_graph.invoke(
                input={"question":question},
                config=config,
            )
            results.append(result)
        
        total_outputs[file_num] = results        
        
    if args.mode == "csv":
        outputs2csv(
            total_outputs=total_outputs, 
            filename=args.file_name
        )
        
    elif args.mode == "eval":
        outputs2pprint(total_outputs)
    
if __name__ == "__main__":   
    def str2bool(value):
        if value.lower() in ('true', '1', 't', 'y', 'yes'):
            return True
        elif value.lower() in ('false', '0', 'f', 'n', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean values expected')
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-fp", "--folder_path", default="./data/input_data/", type=str, help="input data folder path")
    parser.add_argument("-fl", "--file_list", default=[56, 139], nargs='+', type=int, help="input data number")
    parser.add_argument("-rl", "--recursion_limit", default=20, type=int, help="maximum cycle of relevance check recursion")
    parser.add_argument("-fn", "--file_name", default="temp_name", type=str, help="name of csv file")
    parser.add_argument("-m", "--mode", default="csv", choices=["csv", "eval"], type=str, help="choose output form")
    parser.add_argument("-a", "--all_files", default=False, type=str2bool, nargs='?', help="process to all papers")
    args = parser.parse_args()

    main(args)