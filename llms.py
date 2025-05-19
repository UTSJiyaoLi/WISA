import requests
import json

# supprted llm: llama3, gemma3, deepseek-r1
def ollama_summarize(document, model_name="llama3", temperature=0.2):
    """
    Summarize a document using the Ollama server, handling streamed responses.
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": f"Summarize the following text in one sentences:\n\n{document}",
        "options": {
            "temperature": temperature
        },
        "stream": True  # Very important!
    }
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to summarize. Status code: {response.status_code}. Response: {response.text}")

    full_text = ""
    for line in response.iter_lines(decode_unicode=True):
        if line:
            data = json.loads(line)
            full_text += data.get("response", "")
    
    return full_text



def generate_synonyms(word, model_name="llama3", temperature=0.7):
    """
    Generate synonyms for a word using the Ollama server.

    Args:
        word (str): The word you want synonyms for.
        model_name (str): The model name to use (e.g., 'llama3').
        temperature (float): Sampling temperature (higher = more diverse synonyms).

    Returns:
        list: A list of generated synonyms.
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": f"List 5 synonyms for the word '{word}', separated by commas. Only give the words, no explanations.",
        "options": {
            "temperature": temperature
        },
        "stream": True  # Important to handle Ollama streaming output
    }
    
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to generate synonyms. Status code: {response.status_code}. Response: {response.text}")

    full_text = ""
    for line in response.iter_lines(decode_unicode=True):
        if line:
            data = json.loads(line)
            full_text += data.get("response", "")
    
    # Post-process: split the synonyms by commas
    synonyms = [syn.strip() for syn in full_text.split(",") if syn.strip()]
    return synonyms


if __name__ == "__main__":
  # summarize
  doc = """
  Artificial intelligence (AI) has rapidly evolved over the past few years, impacting nearly every sector including healthcare, finance, education, and transportation. Companies and governments are investing heavily in AI research to push the boundaries of what machines can achieve. Despite its promises, AI development raises ethical concerns regarding bias, job displacement, and data privacy. Regulators are now grappling with how to create frameworks that encourage innovation while protecting public interests.
  """

  summary = ollama_summarize(doc, model_name="llama3", temperature=0.3)
  print("Summary:", summary)
  # synonyms
  word = "happy"
  synonyms = generate_synonyms(word, model_name="llama3", temperature=0.7)
  print(synonyms)