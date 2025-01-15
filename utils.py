import os
import time
import yaml
import json
import pprint
import pandas as pd

def load_question(question_path:str="./config/questions/250115-SY-question.yaml"):
    with open(question_path, 'r', encoding="utf-8") as file:
        questions = yaml.safe_load(file)
    
    question_list = []
    for i in range(1, 5):
        if i == 3 or i == 4:
           temp_question = f"""
{questions["main_question"]}{questions[f"add_question{i}"]}
{json.dumps(questions[f"example{i}"], ensure_ascii=False, indent=4)}
""" 
        else: 
            temp_question = f"""
{questions["main_question"]}
{json.dumps(questions[f"example{i}"], ensure_ascii=False, indent=4)}
"""

        question_list.append(temp_question)        

    return question_list


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
        print(f"    JSON 파일 {file_path}에 저장되었습니다.")
        
    else:
        print("    데이터 형식이 지원되지 않습니다. pandas DataFrame 또는 dict만 저장 가능합니다.")


def outputs2csv(total_outputs:dict, filename="temp_result") -> pd.DataFrame:
    answers_list = []
    for file_num in list(total_outputs.keys()):
        outputs = total_outputs[file_num]
        answers = outputs[0]["answer"][0]["CAM (Cathode Active Material)"] | outputs[1]["answer"][0]["Electrode (only for coin-cell (half-cell))"] | outputs[2]["answer"][0]["Morphological results"] | outputs[3]["answer"][0]["Cathode Performance"]
        answers["Paper Number"] = file_num
        answers_list.append(answers)

    answers_csv = pd.DataFrame(answers_list)                               
    columns = ["Paper Number"] + [col for col in answers_csv.columns if col != "Paper Number"]
    answers_csv = answers_csv[columns]
    
    save_data2output_folder(output_folder="./output/csv/", data=answers_csv, filename=filename)
    
    
def outputs2pprint(total_outputs:dict):
    for file_num in list(total_outputs.keys()):
        print(f"    {file_num}번째 논문 결과")
        outputs = total_outputs[file_num]
        answers = outputs[0]["answer"][0] | outputs[1]["answer"][0] | outputs[2]["answer"][0] | outputs[3]["answer"][0]
        pprint.pprint(answers, sort_dicts=False)
        

def outputs2json(total_outputs:dict, file_num:int):
    outputs = total_outputs[file_num]
    answers = outputs[0]["answer"][0] | outputs[1]["answer"][0] | outputs[2]["answer"][0] | outputs[3]["answer"][0]
    
    ## 파일 이름 설정
    if file_num < 10:
        json_file_num = f"00{file_num}"
    elif file_num < 100:
        json_file_num = f"0{file_num}"
    else:
        json_file_num = f"{file_num}"
        
    json_name = f"paper_{json_file_num}_output"
    
    save_data2output_folder(output_folder="./output/json/", data=answers, filename=json_name)