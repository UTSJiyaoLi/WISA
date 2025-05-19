from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from metrics import Metrics
import torch
from typing import List, Tuple
from dataloader import MyDataset
from sum_models import summarize_with_model



def compute_word_importance(model_name: str, doc_tokens: List[str], max_summary_len: int = 128) -> List[Tuple[int, float]]:
    """
    Computes word importance scores based on ROUGE-1 F1 drop using the custom Metrics class.

    Args:
        model_name: HuggingFace summarization model name
        doc_tokens: Tokenized document (list of words)
        max_summary_len: Max length of the generated summary

    Returns:
        List of (index, importance score) sorted by score descending.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()

    metrics = Metrics()

    # 🔥 Summarize original text
    text = " ".join(doc_tokens).strip()
    original_summary = summarize_with_model(text, model_name=model_name)

    scores = []

    for i in range(len(doc_tokens)):
        # 🔥 Delete the i-th word
        modified_tokens = doc_tokens[:i] + doc_tokens[i+1:]
        modified_text = " ".join(modified_tokens).strip()
        modified_summary = summarize_with_model(modified_text, model_name=model_name)

        # 🔥 Compute ROUGE
        rouge_scores = metrics.compute_rouge([modified_summary], [original_summary])
        rouge1_f = rouge_scores['rouge1']  # directly access rouge1

        drop_score = 1.0 - rouge1_f
        scores.append((i, drop_score))

    return sorted(scores, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':

    ds = MyDataset("gigaword", num_of_data=10)
    doc = ds.get_document(0, show_off=False)
    cleaned = ds.clean_text(doc)
    wis = compute_word_importance("t5-base", cleaned)
    print(f'doc: {doc}')
    print(f'wis: {wis}')