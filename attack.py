from transformers import logging
logging.set_verbosity_error()

import argparse
import json
import nltk
import time
import re
import copy
from dataloader import MyDataset
from sum_models import summarize_with_model
from metrics import Metrics
from wis import compute_word_importance
from syn_model import BertSynonymGenerator
from sis import rank_rouge, clean_and_split_sentences
from word_in_sent import rank_words_by_removal
from utils import weighted_sampling
from tqdm import tqdm
from llms import ollama_summarize


def run_attack(args):
    dataset = MyDataset(name=args.dataset, config=args.config, num_of_data=args.num_samples)
    metrics = Metrics()
    model_name = args.model
    synonym_gen = BertSynonymGenerator()

    results = []
    original_summaries, adversarial_summaries, references = [], [], []
    original_docs, modified_docs = [], []

    total_sentence_ranking_time = 0
    total_word_ranking_time = 0
    total_attack_time = 0

    for i in tqdm(range(len(dataset.examples)), desc="Attacking samples"):
        doc = dataset.get_document(i)
        ref_summary = dataset.get_summary(i)
        cleaned_doc = dataset.clean_text(doc)
        cleaned_text = " ".join(cleaned_doc)

        # === Summarize original ===
        if model_name.lower() == "llama3":
            original_summary = ollama_summarize(cleaned_text, model_name="llama3", temperature=0.3)
        else:
            original_summary = summarize_with_model(cleaned_text, model_name)

        original_summaries.append(original_summary)
        references.append(ref_summary)
        original_docs.append(cleaned_text)

        print(f"\n----- Sample {i} -----")
        print("Original summary:", original_summary)

        ### Sentence Ranking ###
        start_time = time.time()
        top_sentences = rank_rouge(dataset, i, method="Pegasus", top_k=args.top_sent_k)
        total_sentence_ranking_time += time.time() - start_time

        ### Word Ranking in top sentences ###
        sentence_word_scores = []
        start_word_ranking = time.time()
        for sent_id, _ in top_sentences:
            word_ranks = rank_words_by_removal(dataset, i, sent_id, ref_summary)
            for word, score in word_ranks:
                sentence_word_scores.append((sent_id, word, score))
        total_word_ranking_time += time.time() - start_word_ranking

        sentence_word_scores = sorted(sentence_word_scores, key=lambda x: x[2], reverse=True)

        ### Sampling-based attack ###
        start_attack = time.time()

        # Initialize best results
        best_adv_summary = original_summary
        best_score_drop = 0
        best_sampled_words = []

        # retake the cleaned sentences
        cleaned_sentences = clean_and_split_sentences(dataset, doc)

        # Initialize best modified sentences as original
        best_modified_sentences = cleaned_sentences[:]

        for t in range(args.T):
            sampled = weighted_sampling(sentence_word_scores, args.sample_size)
            modified_sentences = cleaned_sentences[:]  # Fresh copy each trial
            replaced_words = []

            for sid, word, _ in sampled:
                if sid >= len(modified_sentences):
                    continue

                sentence = modified_sentences[sid]
                sentence_words = nltk.word_tokenize(sentence)

                if word not in sentence_words:
                    continue

                local_idx = sentence_words.index(word)
                synonym = synonym_gen.get_first_synonym(sentence, local_idx)

                if synonym != word:
                    old_sentence = modified_sentences[sid]
                    sentence_words[local_idx] = synonym
                    new_sentence = " ".join(sentence_words)

                    replaced_words.append(
                        (sid, word, synonym, old_sentence, new_sentence)
                    )

                    modified_sentences[sid] = new_sentence

            modified_doc = " ".join(modified_sentences)

            # Summarize modified document
            if model_name.lower() == "llama3":
                modified_summary = ollama_summarize(modified_doc, model_name="llama3", temperature=0.3)
            else:
                modified_summary = summarize_with_model(modified_doc, model_name)

            rouge_diff = metrics.compute_rouge_diff([original_summary], [modified_summary], [ref_summary])
            drop = rouge_diff["rouge1"]

            print(f"Trial {t+1}/{args.T} | ROUGE-1 drop: {drop:.4f}")

            if drop > best_score_drop:
                best_adv_summary = modified_summary
                best_score_drop = drop
                best_sampled_words = copy.deepcopy(replaced_words)
                best_modified_sentences = modified_sentences[:]

        total_attack_time += time.time() - start_attack

        # ✅ Ensure final best document is updated correctly
        best_adv_doc = " ".join(best_modified_sentences)

        adversarial_summaries.append(best_adv_summary)
        modified_docs.append(best_adv_doc)

        results.append({
            "id": i,
            "original_document": cleaned_text,
            "modified_document": best_adv_doc,
            "original_summary": original_summary,
            "adversarial_summary": best_adv_summary,
            "reference_summary": ref_summary,
            "modified_words": [
                {
                    "sentence_id": sid,
                    "original_word": word,
                    "replacement": synonym,
                    "original_sentence": old_sent,
                    "modified_sentence": new_sent
                }
                for sid, word, synonym, old_sent, new_sent in best_sampled_words
            ],
            "final_modified_sentences": best_modified_sentences,
            "rouge_drop": best_score_drop,
            "sentence_ranking_time_sec": round(total_sentence_ranking_time, 2),
            "word_ranking_time_sec": round(total_word_ranking_time, 2),
            "sampling_attack_time_sec": round(total_attack_time, 2),
            "attack_type": "synonym_sampling"
        })

    ### Final Metrics ###
    metric_scores = {
        "rouge_diff": metrics.compute_rouge_diff(original_summaries, adversarial_summaries, references),
        "f1_drop": metrics.compute_f1_diff(original_summaries, adversarial_summaries),
        "semantic_similarity_drop": metrics.compute_similarity_diff(original_summaries, adversarial_summaries),
        "exact_match_drop": metrics.compute_exact_match_diff(original_summaries, adversarial_summaries),
        "avg_modification_rate": metrics.compute_avg_modification_rate(original_docs, modified_docs),
        "avg_original_perplexity": metrics.compute_avg_perplexity(original_summaries),
        "avg_adversarial_perplexity": metrics.compute_avg_perplexity(adversarial_summaries),
        "total_sentence_ranking_time_sec": round(total_sentence_ranking_time, 2),
        "total_word_ranking_time_sec": round(total_word_ranking_time, 2),
        "total_sampling_attack_time_sec": round(total_attack_time, 2),
        "experiment_config": {
            "dataset": args.dataset,
            "config": args.config,
            "model": args.model,
            "num_samples": args.num_samples,
            "top_k": args.top_k,
            "top_sent_k": args.top_sent_k,
            "sample_size": args.sample_size,
            "T": args.T,
            "attack_type": "synonym_sampling"
        }
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(args.metric_json, "w", encoding="utf-8") as f:
        json.dump(metric_scores, f, indent=2)

    print("\n✅ Attack complete.")
    print(f"Results saved to: {args.output_json}")
    print(f"Metrics saved to: {args.metric_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--config", type=str, default="3.0.0")
    parser.add_argument("--model", type=str, default="llama3")  # or google/pegasus-xsum
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_sent_k", type=int, default=3)
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--output_json", type=str, default="results/llama3_atk_outputs_1000.json")
    parser.add_argument("--metric_json", type=str, default="results/llama3_atk_metrics_1000.json")
    args = parser.parse_args()

    run_attack(args)
