import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    PegasusTokenizer, PegasusForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)
# supported models: google/pegasus-xsum, facebook/bart-large-cnn, t5
# --------- Global tokenizer and model (loaded only once) ---------
model_name_global = "google/pegasus-xsum"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "pegasus" in model_name_global.lower():
    tokenizer = PegasusTokenizer.from_pretrained(model_name_global)
    model = PegasusForConditionalGeneration.from_pretrained(model_name_global).to(device)
elif "t5" in model_name_global.lower():
    tokenizer = T5Tokenizer.from_pretrained(model_name_global)
    model = T5ForConditionalGeneration.from_pretrained(model_name_global).to(device)
elif "bart" in model_name_global.lower():
    tokenizer = BartTokenizer.from_pretrained(model_name_global)
    model = BartForConditionalGeneration.from_pretrained(model_name_global).to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name_global)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_global).to(device)

model.eval()

# --------- Summarization function ---------

def summarize_with_model(text: str, model_name: str = None, max_len: int = 128) -> str:
    """
    Summarizes input text using the pre-loaded tokenizer/model.
    """
    if model_name:
        assert model_name == model_name_global, \
            f"Provided model_name ({model_name}) does not match loaded model ({model_name_global})."

    # Preprocess input
    if "t5" in model_name_global.lower():
        input_text = f"summarize: {text.strip()}"
    else:
        input_text = text.strip()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # Input truncation
        padding="longest"
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def batch_summarize_with_model(texts: list, max_len: int = 128, batch_size: int = 8) -> list:
    """
    Summarizes a list of texts in batches to avoid OOM.
    """
    if not isinstance(texts, list):
        raise ValueError("Input must be a list of texts.")

    summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        input_texts = []
        for text in batch:
            if "t5" in model_name_global.lower():
                input_texts.append(f"summarize: {text.strip()}")
            else:
                input_texts.append(text.strip())

        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_len,
                num_beams=4,
                early_stopping=True
            )

        batch_summaries = [tokenizer.decode(sid, skip_special_tokens=True) for sid in summary_ids]
        summaries.extend(batch_summaries)

    return summaries


if __name__ == "__main__":
    # Original document
    original_doc = (
        "A rare white kiwi chick has hatched at a New Zealand wildlife center, "
        "a few days before Christmas. The bird, which has a rare genetic mutation, "
        "was named Manukura. Staff say the chick is healthy and active."
    )

    # Modified document with 2 synonym substitutions
    modified_doc = (
        "A rare white kiwi chick has emerged at a New Zealand sanctuary, "
        "a few days before Christmas. The bird, which has a rare genetic mutation, "
        "was named Manukura. Staff say the chick is healthy and active."
    )

    print("=== Original Document Summary ===")
    original_summary = summarize_with_model(original_doc)
    print(original_summary)

    print("\n=== Modified Document Summary ===")
    modified_summary = summarize_with_model(modified_doc)
    print(modified_summary)
