# Fully Neural Network
import math
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras import Sequential
from keras.preprocessing.text import Tokenizer
from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, LearningRateScheduler, CallbackList, ReduceLROnPlateau

def step_decay(epoch):
    initial_rate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

class FNN():
    def __init__(self, num_classes = 6, w_classes = None):
        self.model = Sequential()
        # 1st layer
        self.model.add(Dense(4096,
                        activation = 'selu', # activation function SELU (Scaled Exponential Linear Unit)
                        kernel_initializer = 'lecun_normal',
                        input_shape = (X_train.shape[1],),
                        kernel_regularizer = tf.keras.regularizers.l2(0.01)))

        # 2nd layer
        self.model.add(Dense(2048,
                        activation = 'selu',
                        kernel_initializer = 'lecun_normal',
                        kernel_regularizer = tf.keras.regularizers.l2(0.01)))

        # 3rd layer
        self.model.add(Dense(1024,
                        activation = 'selu',
                        kernel_initializer = 'lecun_normal',
                        kernel_regularizer = tf.keras.regularizers.l2(0.1)))

        # 4th layer
        self.model.add(Dense(64, activation = 'selu'))

        # 5th layer - output layer
        self.model.add(Dense(num_classes, activation = 'softmax'))

        if w_classes is not None:
            self.model.layers[-1].bias.assign(w_classes)

    def train(self, X_train, y_train, X_val, y_val, num_epochs, batch_size):
        lr_scheduler = LearningRateScheduler(step_decay)
        early_stopping = EarlyStopping(monitor = 'val_loss', restore_best_weights = True, patience = 3)

        return self.model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = num_epochs, batch_size = batch_size, callbacks = [lr_scheduler, early_stopping])

    def predict(self, data, label_encoder = None):
        predict = self.model.predict(data)
        prob_class = predict.argmax(predict, axis = 1)
        if label_encoder is not None:
            return label_encoder.inverse_transform(prob_class)
        return prob_class