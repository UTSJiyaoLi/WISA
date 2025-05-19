# ⚔️ WISA: Weighted Importance Sentence Attack

This repository provides an implementation of **WISA**, an adversarial attack framework designed for text summarization models such as Pegasus, BART, and T5. The core idea is to identify and perturb important input regions—sentences and words—based on ROUGE-driven scoring and guided synonym substitution using large language models (LLMs) like Llama3.

---

## 🔧 Features

- 🧠 **Sentence- and word-level importance ranking** based on ROUGE-1 F1 drop
- 🎯 **Weighted Sampling** to prioritize impactful perturbations
- 🔁 **LLM-based synonym substitution** using locally served models via [Ollama](https://ollama.com)
- 📊 **Comprehensive evaluation metrics**:
  - ROUGE (1/2/L)
  - BLEU
  - Exact Match (EM)
  - F1 Score
  - Semantic Similarity (Sentence-BERT)
  - Perplexity (GPT-2)
  - Modification Rate
- 🧩 **Model-agnostic summarization** via HuggingFace Transformers (Pegasus, BART, T5)

---

## 🛠️ Installation

> Requires Python 3.8+

1. Clone the repository and install dependencies:

bash
pip install -r requirements.txt


2. Download NLTK sentence tokenizer:

bash
python -m nltk.downloader punkt


---

## 🧠 LLM Setup (via Ollama)

WISA leverages local LLMs (e.g., Llama3) via [Ollama](https://ollama.com).

### Step 1: Install Ollama

Visit: [https://ollama.com](https://ollama.com)  
Follow platform-specific instructions to install the Ollama runtime.

### Step 2: Start Ollama server and pull model

bash
ollama serve          # or ./ollama serve
ollama pull llama3    # download Llama3 locally


### Step 3: Run Llama3 model

bash
ollama run llama3


Alternatively, run from Python:

bash
python llms.py


---

## 🚀 Run WISA Attack

To launch the attack:

bash
python attack.py


- Make sure to edit attack.py to adjust hyperparameters such as:
  - top_k sentences
  - LLM model name (llama3)
  - sampling size
  - batch size, etc.

---

## 🧪 Example Usage

You may also test individual modules directly:

- **Sentence importance scoring:**

```
bash
python sis.py
```

- **Word importance analysis:**

```
bash
python wis.py
```

- **Word substitution and synonym generation:**

```
bash
python syn_model.py
```

- **Metric evaluation preview:**

```
bash
python metrics.py
```


---

## 🤝 Acknowledgments

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Sentence-Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.com) for local LLM support
- [GPT-2](https://huggingface.co/gpt2) for Perplexity calculation
- [BERT](https://huggingface.co/bert-base-uncased) for synonym replacement

---

## 📬 Contact

If you find this work helpful or would like to contribute, feel free to open an issue or pull request.
