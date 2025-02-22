# meteor_metric.py
import nltk
nltk.download("wordnet")

from nltk.translate import meteor_score
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer
from langchain_teddynote.community.kiwi_tokenizer import KiwiTokenizer



def calculate_meteor(question, answer):
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    wn.ensure_loaded()
    meteor = meteor_score.meteor_score(
    [tokenizer.tokenize(question)],
    tokenizer.tokenize(answer),
    )
    
    return meteor