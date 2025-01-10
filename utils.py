import os
import time
import pprint
import pandas as pd

def save_csv2output_folder(output_folder: str, df: pd.DataFrame, filename: str):
    # output 폴더 경로
    output_folder = output_folder
    
    # output 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 파일 이름 설정
    timestamp = time.strftime("%y%m%d%H%M%S")
    filename = timestamp + "-" + filename + ".csv"
    
    # CSV 파일 저장 경로
    output_path = os.path.join(output_folder, filename)
    
    # DataFrame을 CSV로 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"{filename} 파일이 {output_path}에 저장되었습니다.")


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
    
    save_csv2output_folder(output_folder="./output/", df=answers_csv, filename=filename)
    
    
def outputs2pprint(total_outputs:dict):
    for file_num in list(total_outputs.keys()):
        print(f"##  {file_num}번째 논문 결과")
        outputs = total_outputs[file_num]
        answers = outputs[0]["answer"][0] | outputs[1]["answer"][0] | outputs[2]["answer"][0] | outputs[3]["answer"][0]
        pprint.pprint(answers)