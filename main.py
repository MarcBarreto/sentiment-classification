import os
from fnn import FNN
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

    model = FNN(num_classes = 6, w_classes = w_classes)
    history = model.train(X,train, y_train_encoded, X_val, y_val_encoded, num_epochs = 30, batch_size = 256)

    predict = model.predict(X_test_array)
    print(f'Classification Report: {Classification_report(y_test_le, predict)}')

    print(f'Confusion matrix: {Confusion_matrix(y_test_le, predict)}')

    model.save('./model_v1.keras')

    # Deploy
    text = ' '
    while(text !== quit) {
        text = input('Type the sentence to be analyzed or quit to exit')
        df = pd.DataFrame({'Phrase': [text]})

        df['Processed_Phrase'] = df['Phrase'].apply(utils.preprocess_text)

        df_tfidf = utils.tfidf_vectorizer(df['Processed_Phrase'], fit = True, tfidf = tfidf)
        df_tfidf_array = df_tfidf.toarray()

        result = model.predict(df_tfidf_array)

        prob_class = np.argmax(result, axis = 1)
        class_name = label_encoder.inverse_transform(prob_class)
        print(f'The sentiment is {class_name}')
    }