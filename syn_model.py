from transformers import BertForMaskedLM, BertTokenizer
import torch
import nltk


class BertSynonymGenerator:
    def __init__(self, model_name="bert-base-uncased", top_k=10):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).eval()
        self.top_k = top_k

    def get_first_synonym(self, sentence, word_idx):
        """
        Replace the word at word_idx in the sentence with a predicted synonym.
        Args:
            sentence: A single sentence (string), already cleaned (only alphabetic words).
            word_idx: Index of the word in the sentence.
        Returns:
            A synonym string. If no valid synonym is found, return the original word.
        """

        words = nltk.word_tokenize(sentence)
        if word_idx >= len(words):
            return words[word_idx]  # No change if index out of bounds

        original_word = words[word_idx]

        # Mask the target word
        masked_words = words[:]
        masked_words[word_idx] = self.tokenizer.mask_token
        masked_sentence = " ".join(masked_words)

        # Tokenize
        inputs = self.tokenizer(masked_sentence, return_tensors="pt")

        # Check if input length is safe
        if inputs["input_ids"].shape[1] > 512:
            print(f"[Warning] Sentence too long after masking: {inputs['input_ids'].shape[1]} tokens.")
            return original_word  # Skip replacement

        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        # Find the position of [MASK]
        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        # Get top-k predicted tokens for the masked position
        mask_logits = logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_logits, self.top_k, dim=1).indices[0].tolist()

        for token_id in top_tokens:
            synonym = self.tokenizer.decode([token_id]).strip()
            # Skip the original word and non-alphabetic predictions
            if synonym.lower() != original_word.lower() and synonym.isalpha():
                return synonym

        return original_word  # Fallback if no suitable synonym found



if __name__ == "__main__":
    sgen = BertSynonymGenerator()
    sent = "The capital of France is Paris"
    index = 3  # replace "France"
    synonym = sgen.get_first_synonym(sent, index)
    print(f"Synonym for word {index}: {synonym}")
