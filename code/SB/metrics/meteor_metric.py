# meteor_metric.py
from nltk.translate import meteor_score
from transformers import AutoTokenizer

# BERT 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(text):
    """Hugging Face BERT 토크나이저를 사용한 토큰화"""
    return tokenizer.tokenize(text)

def calculate_meteor(question, answer):
  
  
    reference_tokens = [tokenize(question)]  # 토큰화된 리스트
    candidate_tokens = [tokenize(answer) ]   # 토큰화된 리스트

    
    meteor = meteor_score.meteor_score(reference_tokens, candidate_tokens)
    return meteor