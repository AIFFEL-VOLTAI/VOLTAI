"""
자동요약 및 기계 번역의 품질을 평가하는 데 사용되는 평가지표
생성된 텍스트가 참조 텍스트의 중요 키워드를 얼마나 포함하고 있는지 측정함
n-gram 중첩을 기반으로 계산한다.

Rouge-1 - 단어 단위의 유사도를 측정합니다. - 두 문장간의 개별 단어 일치도를 평가합니다.

Rouge-2 - 두 단어 연속(bigram)의 중복 단위의 유사도를 측정합니다. - 두 문장간의 연속된 두 단어 일치도를 평가합니다.

Rouge-L - 최장 공통 부분 수열(Longest Common Subsequence, LCS)을 기반으로 한 유사도를 측정합니다. 
- 문장 수준의 단어 순서를 고려하며, 연속적인 일치를 요구하지 않습니다 
- 더 유연한 평가가 가능하며, 문장 구조의 유사성을 자연스럽게 반영합니다.

"""


""""""

# rouge_metric.py
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

# 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(text):
    return tokenizer.tokenize(text)

def calculate_rouge(question, answer):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(' '.join(tokenize(question)), ' '.join(tokenize(answer)))
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }
