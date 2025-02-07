import yaml

def question_generator(
    question_path: str = "./configs/questions/multiagent.yaml", 
    category_number: int = None, 
    sample_names: list = None) -> str:
    
    ## question template 불러오기
    with open(question_path, 'r', encoding="utf-8") as file:
        question = yaml.safe_load(file)

    print(f"##  {question_path} template를 불러왔습니다.")
    
    ## category 별 question 생성
    category_names = ["CAM (Cathode Active Material)", "Electrode (half-cell)", "Morphological Properties", "Cathode Performance"]
    
    for i, sample_name in enumerate(sample_names):
        if category_number == 1:
            question[f"category{category_number}"]["template"][category_names[category_number-1]]["Stoichiometry information"][sample_name] = {}
            question[f"category{category_number}"]["template"][category_names[category_number-1]]["Commercial NCM used"][sample_name] = {}
    
        elif category_number == 3:
            temp_template = question[f"category{category_number}"]["template"][category_names[category_number-1]]
            for k in temp_template.keys():
                question[f"category{category_number}"]["template"][category_names[category_number-1]][k][sample_name] = None
    
        elif category_number == 4:
            temp_performance = question[f"category{category_number}"]["template"]["Cathode Performance"][""]
            question[f"category{category_number}"]["template"]["Cathode Performance"].update({sample_name:temp_performance})
            if i == len(sample_names)-1:
                del question[f"category{category_number}"]["template"]["Cathode Performance"][""]
        
        print(f"    - {sample_name}에 대한 question을 생성했습니다.")

    ## total question 생성
    question_text = question[f"category{category_number}"]['question_text']
    template = question[f"category{category_number}"]['template'] 
    
    return f"{question_text}{[template]}".replace("'", '"')