import os
import pandas as pd
import utils

if __name__ == '__main__':
    train_data = pd.read_csv('./datas/dados_treino.txt', header = None, delimiter = ';')
    test_data = pd.read_csv('./datas/dados_teste.txt', header = None, delimiter = ';')

    train_data.rename(columns = {0: 'text', 1: 'sentiment'})
    test_data.rename(columns = {0: 'text', 1: 'sentiment'})

    train_data['processed_text'] = train_data['text'].apply(utils.preprocess_text)
    test_data['processed_text'] = test_data['text'].apply(utils.preprocess_text)