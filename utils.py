import spacy
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

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

def tfidf_vectorizer(data, fit = False, tfidf = None, max_df = 0.95, min_df = 2, stop_words = 'english')
    if fit and tfidf is not None:
        return tfidf.transform(data)
    tfidf = TfidfVectorizer(max_df = max_df, min_df = min_df, stop_words = stop_words)
    data = tfid.fit_transform(data)
    return data, tfidf

def apply_label_encoder(data, fit = False, label_encoder = None)
    if fit and label_encoder is not None:
        return label_encoder.transform(data)
    label_encoder = LabelEncoder()
    data_encoded = label_encoder.fit_transform(data)
    return data_encoded, label_encoder

def weigh_class(data):
    return compute_class_weigh('balanced', classes = np.unique(data), y = data)
