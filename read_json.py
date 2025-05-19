import json
import argparse

def load_results(result_json):
    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_metrics(metric_json):
    with open(metric_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics

def print_sample_results(results):
    for sample in results:
        print("\n==============================")
        print(f"Sample ID: {sample['id']}")
        print("------------------------------")
        print(f"Original Summary:\n{sample['original_summary']}\n")
        print(f"Adversarial Summary:\n{sample['adversarial_summary']}\n")
        print(f"Modified Words: {sample['modified_words']}")
        print(f"ROUGE Drop: {sample['rouge_drop']}")
        print("------------------------------")
        print(f"Sentence Ranking Time (sec): {sample['sentence_ranking_time_sec']}")
        print(f"Word Ranking Time (sec): {sample['word_ranking_time_sec']}")
        print(f"Sampling Attack Time (sec): {sample['sampling_attack_time_sec']}")
        print("==============================\n")

def print_overall_metrics(metrics):
    print("\n===== Experiment Configuration =====")
    config = metrics.get("experiment_config", {})
    for key, value in config.items():
        print(f"{key}: {value}")
    print("===========================\n")
    print("\n===== Overall Metrics =====")
    print("ROUGE Difference:", metrics.get("rouge_diff"))
    print("F1 Drop:", metrics.get("f1_drop"))
    print("Semantic Similarity Drop:", metrics.get("semantic_similarity_drop"))
    print("Exact Match Drop:", metrics.get("exact_match_drop"))
    print("Average Modification Rate:", metrics.get("avg_modification_rate"))
    print("Average Original Perplexity:", metrics.get("avg_original_perplexity"))
    print("Average Adversarial Perplexity:", metrics.get("avg_adversarial_perplexity"))
    print("\nTotal Sentence Ranking Time (sec):", metrics.get("total_sentence_ranking_time_sec"))
    print("Total Word Ranking Time (sec):", metrics.get("total_word_ranking_time_sec"))
    print("Total Sampling Attack Time (sec):", metrics.get("total_sampling_attack_time_sec"))
    print("===========================\n")    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", type=str, default="results/llama3_atk_outputs_500.json", help="Attack result JSON file")
    parser.add_argument("--metric_json", type=str, default="results/llama3_atk_metrics_500.json", help="Attack metrics JSON file")
    args = parser.parse_args()

    results = load_results(args.result_json)
    metrics = load_metrics(args.metric_json)

    print_sample_results(results)
    print_overall_metrics(metrics)
    