import subprocess

def load_dict(dict):
    command = f'python -m spacy download {dict}'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    nlp_dict = spacy.load(dict)
    return nlp_dict

def preprocess_text(text, nlp_dict):
    doc = nlp_dict(text)
    tokens = [tokens.lemma.lower().strip() for token in doc if not token.is_stop]
    return ' '.join(tokens)