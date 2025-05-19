import re
import nltk
from datasets import load_dataset
from typing import Optional, Tuple, List, Union


class MyDataset:
    def __init__(
        self,
        name,
        split="test",
        config=None,
        num_of_data=10,
    ):
        self.name = name
        self.config = config
        self.split = split
        # 🔄 Use streaming to avoid full download
        self.dataset = load_dataset(self.name, self.config, split=self.split, streaming=True, trust_remote_code=True)
        self.examples = [example for _, example in zip(range(num_of_data), self.dataset)]

    def get_document(self, index: int, show_off: bool = False) -> str:
        """
        Returns the source document text from the dataset.
        """
        field = {
            'cnn_dailymail': 'article',
            'multi_news': 'document',
            'xsum': 'document',
            'samsum': 'dialogue',
            'gigaword': 'document'
        }.get(self.name, 'document')

        doc = self.examples[index][field]
        if show_off:
            print(f"\033[32m{doc.replace(chr(10), ' ')}\033[0m")

        return doc

    def get_summary(self, index: int) -> str:
        """
        Returns the summary text.
        """
        field = 'highlights' if self.name == 'cnn_dailymail' else 'summary'
        return self.examples[index][field]


    def clean_text(self, text: str) -> List[str]:
        """
        Cleans input text and returns a list of pure word tokens (alphabetic only).
        Filters out punctuation, numbers, and symbols.
        """
        # Normalize whitespace and remove HTML breaks
        text = re.sub(r'<br\s*/?>', '.', text)
        text = re.sub(r'\s+', ' ', text.strip())

        # Tokenize into words
        tokens = nltk.word_tokenize(text)

        # Keep only alphabetic words
        words = [word for word in tokens if word.isalpha()]

        return words

        
    def get_gap_document(self, index: int, sentence_id: int, method: Optional[str] = None) -> Tuple[str, int]:
        """
        Removes the specified sentence (by sentence_id) from the document.
        Returns the modified document and the removed sentence index.
        """
        document = self.get_document(index)
        if self.name == "xsum":
            sentences = re.split(r"\n", document)
        else:
            sentences = nltk.sent_tokenize(document)

        if 0 <= sentence_id < len(sentences):
            removed_sentence = sentences.pop(sentence_id)
        else:
            removed_sentence = ""

        modified_doc = " ".join(sentences)

        return modified_doc, sentence_id 


    def remove_word_by_index(self, document, index):
        """
        Cleans the document, removes the word at the specified index (if valid),
        and returns the modified document as a string.
        """
        words = self.clean_text(document)  # Call another method from the same class
        if 0 <= index < len(words):
            del words[index]
        else:
            print(f"Index {index} is out of bounds. No word was removed.")
        return ' '.join(words)


    def threshold(self, return_range: bool = False) -> Union[float, Tuple[float, int, int]]:
        """
        Computes average, min, and max number of words (after cleaning) per document.
        """
        lens = []
        for i in range(len(self.examples)):
            doc = self.get_document(i)
            words = self.clean_text(doc)
            lens.append(len(words))

        avg_len = sum(lens) / len(lens)
        return (avg_len, min(lens), max(lens)) if return_range else avg_len



if __name__ == "__main__":
    # test dataset
    # ds = MyDataset("cnn_dailymail", config="3.0.0", num_of_data=10)
    # ds = MyDataset("xsum", num_of_data=10)
    # ds = MyDataset("multi_news", num_of_data=10)
    # ds = MyDataset("samsum", num_of_data=10)
    ds = MyDataset("gigaword", num_of_data=10)
    doc = ds.get_document(0, show_off=False)
    print(doc)

    summary = ds.get_summary(0)
    print(f'summary: {summary}')

    cleaned = ds.clean_text(doc)
    print(f'cleaned: {cleaned}')

    avg = ds.threshold(return_range=True)
    # print(f'avg: {avg}')
    word_index_to_remove = 5

    modified_doc = ds.remove_word_by_index(doc, word_index_to_remove)

    print("\nModified Document (after removing word at index {}):".format(word_index_to_remove))
    print(modified_doc)