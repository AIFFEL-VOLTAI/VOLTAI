# semantic_similarity_metric.py
from sentence_transformers import SentenceTransformer, util
import warnings

# FutureWarning 제거
warnings.filterwarnings("ignore", category=FutureWarning)

# Sentence Transformer 모델 로드
model = SentenceTransformer("all-mpnet-base-v2")

def calculate_semantic_similarity(question, answer):
    """Sentence-BERT를 사용한 Semantic Textual Similarity (STS) 점수 계산"""
    question_encoded = model.encode(question, convert_to_tensor=True)
    answer_encoded = model.encode(answer, convert_to_tensor=True)
    score = util.pytorch_cos_sim(question_encoded, answer_encoded).item()
    return score