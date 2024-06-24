import os
import utils
import pandas as pd
from keras.utils import to_categorical

if __name__ == '__main__':
    # Loading data
    train_data = pd.read_csv('./datas/dados_treino.txt', header = None, delimiter = ';')
    test_data = pd.read_csv('./datas/dados_teste.txt', header = None, delimiter = ';')

    train_data.rename(columns = {0: 'text', 1: 'sentiment'})
    test_data.rename(columns = {0: 'text', 1: 'sentiment'})

    # Pre Processing text with spacy
    train_data['processed_text'] = train_data['text'].apply(utils.preprocess_text)
    test_data['processed_text'] = test_data['text'].apply(utils.preprocess_text)

    # Create Vectorizer (Tfidf)
    tfidf_train_data, tfidf = utils.tfidf_vectorizer(train_data['processed_text'])
    tfidf_test_data = utils.tfidf_vectorizer(test_data['processed_text'], fit = True, tfidf = tfidf)

    X_train_array = tfidf.train_data.toarray()
    X_test_array = tfidf.test_data.toarray()

    y_train_le, label_encoder = utils.apply_label_encoder(train_data['sentiment'])
    y_test_le = utils.apply_label_encoder(test_data['sentiment'], fit = True, label_encoder = label_encoder)

    w_classes = utils.weigh_class(y_train_le)

    X_train, X_val, y_train, y_val = train_test_split(X_train_array, y_train_le, test_size = 0.2, random_state = 42, stratify = y_train_le)

    y_train_encoded = to_categorical(y_train)
    y_val_encoded = to_categorical(y_Val)
    y_test_encoded = to_categorical(y_test_le)