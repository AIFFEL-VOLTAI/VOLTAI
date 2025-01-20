# evaluate_all.py
from .rouge_metric import calculate_rouge
from .bleu_metric import calculate_bleu
from .meteor_metric import calculate_meteor
from .sem_score_metric import calculate_semantic_similarity

def evaluate_all_metrics(question, answer):
    results = {
        "ROUGE": calculate_rouge(question, answer),
        "BLEU": calculate_bleu(question, answer),
        "METEOR": calculate_meteor(question, answer),
        "Semantic Similarity (STS)": calculate_semantic_similarity(question, answer)
    }
    return results

