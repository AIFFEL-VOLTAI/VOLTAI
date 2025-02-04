# bleu_metric.py
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer

# 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(text):
    return tokenizer.tokenize(text)

def calculate_bleu(question, answer):
    reference = [tokenize(question)]
    candidate = tokenize(answer)
    print("Success BLEU")
    return sentence_bleu(reference, candidate)

