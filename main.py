import os
import utils
import transformers
import pandas as pd
from fnn import FNN
from tensorflow import keras
from keras.utils import to_categorical
from transformers import TFDistilBertModel
from tokenizers import BertWordPieceTokenizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.preprocessing.sequence import pad_sequences

def encode(texts, tokenizer, chunk_size = 256, maxlen = 512):
  # Enable truncation in tokenizer to max lenght
  tokenizer.enable_truncation(max_length = maxlen)

  # Enable padding in tokenizer
  tokenizer.enable_padding(length = maxlen)

  all_ids = []

  # Iterate over all text in 'chunk pieces'
  for i in tqdm(range(0, len(texts), chunk_size)):
    text_chunk = texts[i:i+chunk_size].tolist()

    encs = tokenizer.encode_batch(text_chunk)

    # Extemded the list 'all_ids' with encoded Ids
    all_ids.extended(enc.ids for enc in encs)

  return np.array(all_ids)

if __name__ == '__main__':
    print(f'Waiting! Loading models...')

    # Loading model
    try:
        fnn_model = keras.models.load_model('./models/model_v1.keras')
        lstm_model = keras.models.load_model('./models/model_v2.keras')
        with custom_object_scope({'TFDistilBertModel': TFDistilBertModel}):
            transformer_model = tf.keras.models.load_model('./models/model_v3.keras')
    except:
        print('Error to load models')

    # Loading data
    train_data = pd.read_csv('./datas/dados_treino.txt', header = None, delimiter = ';')

    train_data.rename(columns = {0: 'text', 1: 'sentiment'})

    # Pre Processing text with spacy
    train_data['processed_text'] = train_data['text'].apply(utils.preprocess_text)

    # Create Vectorizer (Tfidf)
    tfidf_train_data, tfidf = utils.tfidf_vectorizer(train_data['processed_text'])

    X_train_array = tfidf.train_data.toarray()

    y_train_le, label_encoder = utils.apply_label_encoder(train_data['sentiment'])

    # LSTM Tokenizer
    lstm_tokenizer = Tokenizer()
    lstm_tokenizer.fit_on_texts(train_data['Processed_text'])

    # Transformer Tokenizer
    bert_tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    tokenizer_bert.save_pretrained('.')
    fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase = False)

    max_length = 100
    # Deploy
    text = ' '
    while(text !== quit) {
        text = input('Type the sentence to be analyzed or quit to exit')
        df = pd.DataFrame({'Phrase': [text]})

        df['Processed_Phrase'] = df['Phrase'].apply(utils.preprocess_text)
        
        # FNN Inference
        df_tfidf = utils.tfidf_vectorizer(df['Processed_Phrase'], fit = True, tfidf = tfidf)
        df_tfidf_array = df_tfidf.toarray()

        fnn_result = model.predict(df_tfidf_array)

        fnn_prob_class = np.argmax(fnn_result, axis = 1)
        fnn_class_name = label_encoder.inverse_transform(fnn_prob_class)

        # LSTM Inference
        new_seq = lstm_tokenizer.texts_to_sequences(df['Processed_phrase'])
        new_seq_pad = pad_sequences(new_seq, maxlen = max_length)

        lstm_result = lstm_model.predict(new_seq_pad)

        lstm_prob_class = np.argmax(lstm_result, axis = 1)
        lstm_class_name = label_encoder.inverse_transform(lstm_prob_class)

        # Transformer (Bert) Inference
        new_data = encode(df['Processed_phrase'], fast_tokenizer, max_len = max_length)
        
        transformer_result = transformer_model.predict(new_data)
        
        transformer_prob_class = np.argmax(transformer_result, axis = 1)
        transformer_class_name = label_encoder.inverse_transform(transformer_prob_class)

        # Result    
        print(f'Result of FNN Model: Sentiment: {fnn_class_name}. Score: {fnn_prob_class:.2f}')
        print(f'Result of LSTM Model: Sentiment: {lstm_class_name}. Score: {lstm_prob_class:.2f}')
        print(f'Result of Transformer (Bert) Model: Sentiment: {transformer_class_name}. Score: {transformer_prob_class:.2f}')
    }