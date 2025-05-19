from transformers import logging
logging.set_verbosity_error()

import nltk
import re
from dataloader import MyDataset
from sum_models import summarize_with_model
from metrics import Metrics


def clean_and_split_sentences(dataset, document):
    """
    Clean the document and split into cleaned sentences.
    Always use this for sentence-level operations to ensure consistency.
    """
    if dataset.name == "xsum":
        raw_sentences = re.split(r"\n", document)
    else:
        raw_sentences = nltk.sent_tokenize(document)

    cleaned_sentences = []
    for sent in raw_sentences:
        sent_clean = " ".join(dataset.clean_text(sent))
        if sent_clean.strip(): 
            cleaned_sentences.append(sent_clean)

    return cleaned_sentences

def rank_rouge(dataset, doc_id, method="Pegasus", top_k=3):
    """
    Rank sentences in a document by ROUGE score drop when each sentence is removed.
    """
    rank_model = "google/pegasus-xsum"
    metrics = Metrics()

    document = dataset.get_document(doc_id)
    reference_summary = dataset.get_summary(doc_id)

    # clean sentences
    cleaned_sentences = clean_and_split_sentences(dataset, document)
    cleaned_document = " ".join(cleaned_sentences)

    if len(cleaned_sentences) == 0:
        return []

    original_summary = summarize_with_model(cleaned_document, model_name=rank_model)

    sentence_scores = []

    for i in range(len(cleaned_sentences)):
        modified_sentences = cleaned_sentences[:]
        modified_sentences[i] = "" 
        modified_doc = " ".join(modified_sentences)

        modified_summary = summarize_with_model(modified_doc, model_name=rank_model)

        rouge_diff = metrics.compute_rouge_diff(
            [original_summary], [modified_summary], [reference_summary]
        )
        drop = rouge_diff["rouge1"]
        sentence_scores.append((i, drop))

    # Sentence Importance Ranking (SIR)
    sentence_scores = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    top_sentences = []
    used = set()
    for sid, drop in sentence_scores:
        if sid not in used:
            top_sentences.append((sid, drop))
            used.add(sid)
        if len(top_sentences) >= top_k:
            break

    return top_sentences

def get_sentence(dataset, doc_id, sent_id):
    document = dataset.get_document(doc_id)
    if dataset.name == "xsum":
        sentences = re.split(r"\n", document)
    else:
        sentences = nltk.sent_tokenize(document)
    return sentences[sent_id] if sent_id < len(sentences) else ""

if __name__ == '__main__':
    ds = MyDataset("cnn_dailymail", config="3.0.0", num_of_data=5)
    doc_id = 0

    top_sentences = rank_rouge(ds, doc_id, method="Pegasus", top_k=5)

    print("Top-5 important sentences:")
    for sid, score in top_sentences:
        sent = get_sentence(ds, doc_id, sid)
        print(f"Sentence {sid}: {sent} (ROUGE-1 drop: {score:.4f})")
