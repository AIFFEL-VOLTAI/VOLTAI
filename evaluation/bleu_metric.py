# bleu_metric.py
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer

# 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

def tokenize(text):
    return tokenizer.tokenize(text)

def calculate_bleu(question, answer):
    reference = [tokenize(question)]
    candidate = tokenize(answer)
    return sentence_bleu(reference, candidate)

