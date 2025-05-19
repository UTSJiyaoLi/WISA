import re
import torch
import nltk
from typing import List
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class Metrics:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        self.ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def compute_rouge(self, preds: List[str], refs: List[str]) -> dict:
        rouge1_f1, rouge2_f1, rougel_f1 = [], [], []
        for pred, ref in zip(preds, refs):
            scores = self.rouge.score(ref, pred)
            rouge1_f1.append(scores['rouge1'].fmeasure)
            rouge2_f1.append(scores['rouge2'].fmeasure)
            rougel_f1.append(scores['rougeL'].fmeasure)
        return {
            'rouge1': sum(rouge1_f1) / len(rouge1_f1),
            'rouge2': sum(rouge2_f1) / len(rouge2_f1),
            'rougeL': sum(rougel_f1) / len(rougel_f1)
        }

    def compute_rouge_diff(self, orig_summaries, adv_summaries, refs):
        orig = self.compute_rouge(orig_summaries, refs)
        adv = self.compute_rouge(adv_summaries, refs)
        return {
            'rouge1': orig['rouge1'] - adv['rouge1'],
            'rouge2': orig['rouge2'] - adv['rouge2'],
            'rougeL': orig['rougeL'] - adv['rougeL']
        }

    def compute_bleu(self, pred: str, ref: str) -> float:
        reference_tokens = [ref.lower().split()]
        prediction_tokens = pred.lower().split()
        bleu1 = sentence_bleu(reference_tokens, prediction_tokens, weights=(1.0, 0, 0, 0))
        return bleu1

    def compute_exact_match(self, pred: str, ref: str) -> float:
        pred_words = set(pred.strip().lower().split())
        ref_words = set(ref.strip().lower().split())
        overlap = pred_words.intersection(ref_words)
        return len(overlap) / len(ref_words) if ref_words else 0.0

    def compute_f1(self, pred: str, ref: str) -> float:
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        common_tokens = set(pred_tokens) & set(ref_tokens)
        if not common_tokens:
            return 0.0
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def compute_f1_diff(self, originals, adversarials):
        f1_orig = []
        f1_adv = []
        for o, a in zip(originals, adversarials):
            f1_orig.append(self.compute_f1(o, a))
            f1_adv.append(0.0)  
        return sum(f1_orig) / len(f1_orig) - sum(f1_adv) / len(f1_adv)

    def compute_similarity(self, pred: str, ref: str) -> float:
        emb1 = self.sim_model.encode([pred])[0]
        emb2 = self.sim_model.encode([ref])[0]
        return float(cosine_similarity([emb1], [emb2])[0][0])

    def compute_similarity_diff(self, originals, adversarials):
        sims_orig = []
        sims_adv = []
        for o, a in zip(originals, adversarials):
            sims_orig.append(self.compute_similarity(o, o))
            sims_adv.append(self.compute_similarity(o, a))
        return sum(sims_orig) / len(sims_orig) - sum(sims_adv) / len(sims_adv)

    def compute_exact_match_diff(self, originals, adversarials):
        em_orig = []
        em_adv = []
        for o, a in zip(originals, adversarials):
            em_orig.append(1.0)
            em_adv.append(self.compute_exact_match(o, a))
        return sum(em_orig) / len(em_orig) - sum(em_adv) / len(em_adv)

    def compute_avg_modification_rate(self, originals, modifieds):
        rates = []
        for o, m in zip(originals, modifieds):
            o_words = o.split()
            m_words = m.split()
            diff = sum(1 for a, b in zip(o_words, m_words) if a != b)
            rates.append(diff / max(len(o_words), 1))
        return sum(rates) / len(rates)

    def compute_perplexity(self, text: str) -> float:
        tokens = self.ppl_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = self.ppl_model(**tokens, labels=tokens["input_ids"])
        return float(torch.exp(output.loss))

    def compute_avg_perplexity(self, texts):
        perplexities = []
        for text in texts:
            try:
                perplexities.append(self.compute_perplexity(text))
            except Exception:
                perplexities.append(1000.0)
        return sum(perplexities) / len(perplexities)

if __name__ == '__main__':
    # Initialize the metrics class
    metrics = Metrics()

    # Example texts
    ref = "The capital of France is Paris"
    pred = "Paris is the capital of France"

    # ROUGE (batch input required)
    rouge_scores = metrics.compute_rouge([pred], [ref])
    print(f"ROUGE Scores: {rouge_scores}")

    # BLEU
    bleu_score = metrics.compute_bleu(pred, ref)
    print(f"BLEU-1 Score: {bleu_score:.4f}")

    # F1 and Exact Match (EM)
    f1_score = metrics.compute_f1(pred, ref)
    em_score = metrics.compute_exact_match(pred, ref)
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Exact Match (Soft Overlap) Score: {em_score:.4f}")

    # Semantic Similarity
    similarity_score = metrics.compute_similarity(pred, ref)
    print(f"Semantic Similarity: {similarity_score:.4f}")

    # Modification Rate (difference between texts)
    mod_rate = metrics.compute_modification_rate(ref, pred)
    print(f"Modification Rate: {mod_rate:.4f}")

    # Perplexity (how natural the generated prediction is)
    perplexity_score = metrics.compute_perplexity(pred)
    print(f"Perplexity: {perplexity_score:.2f}")