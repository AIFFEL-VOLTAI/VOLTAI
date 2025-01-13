# meteor_metric.py
from nltk.translate import meteor_score
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer
from langchain_teddynote.community.kiwi_tokenizer import KiwiTokenizer



def calculate_meteor(question, answer):
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    wn.ensure_loaded()
    meteor = meteor_score.meteor_score(
    [tokenizer.tokenize(question)],
    tokenizer.tokenize(answer),
    )
    return meteor