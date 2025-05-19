from sum_models import summarize_with_model, batch_summarize_with_model
from rouge_score import rouge_scorer
import nltk, re
from sis import get_sentence, rank_rouge
from dataloader import MyDataset
import time

def rank_words_by_removal(dataset, doc_id, sentence_id, original_summary, model_name="google/pegasus-xsum"):
    """
    For a given sentence, delete each word and compute ROUGE drop to rank word importance.
    Returns: List of (word, rouge_drop)
    """
    document = dataset.get_document(doc_id)

    if dataset.name == "xsum":
        raw_sentences = re.split(r"\n", document)
    else:
        raw_sentences = nltk.sent_tokenize(document)

    cleaned_sentences = []
    for sent in raw_sentences:
        words = dataset.clean_text(sent)
        cleaned_sentence = " ".join(words)
        cleaned_sentences.append(cleaned_sentence)

    cleaned_document = " ".join(cleaned_sentences)
    tgt_text_raw = summarize_with_model(cleaned_document, model_name=model_name)

    # Prepare the target sentence
    target_sentence = cleaned_sentences[sentence_id]
    words = nltk.word_tokenize(target_sentence)

    # Prepare modified documents by removing each word
    modified_docs = []
    for idx in range(len(words)):
        modified_words = words[:idx] + words[idx+1:]
        modified_sentence = " ".join(modified_words)
        modified_sentences = cleaned_sentences[:]
        modified_sentences[sentence_id] = modified_sentence
        modified_doc = " ".join(modified_sentences)
        modified_docs.append(modified_doc)

    # Batch summarize all modified versions
    modified_summaries = batch_summarize_with_model(modified_docs)

    # Compute ROUGE drop
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores_raw = scorer.score(original_summary, tgt_text_raw)

    word_scores = []
    for idx, modified_summary in enumerate(modified_summaries):
        scores_mod = scorer.score(original_summary, modified_summary)
        rouge_drop = scores_raw["rouge1"].fmeasure - scores_mod["rouge1"].fmeasure
        word_scores.append((words[idx], rouge_drop))

    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
    return word_scores


if __name__  == '__main__':

    ds = MyDataset("cnn_dailymail", config="3.0.0", num_of_data=5)
    doc_id = 0
    top_sentences = rank_rouge("cnn_dailymail", ds, doc_id, method="Pegasus", top_k=3)
    original_summary = ds.get_summary(doc_id)
    start_time = time.time()

    for sent_id, score in top_sentences:
        print(f"\nSentence {sent_id} (score: {score:.4f})")
        word_ranks = rank_words_by_removal(ds, doc_id, sent_id, original_summary, model_name="google/pegasus-xsum")
        for word, word_score in word_ranks:
            print(f"  Word: {word}   Importance: {word_score:.4f}")
    end_time = time.time()
    print(f'Time consumption: {end_time - start_time}')
