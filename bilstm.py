import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
from tensorflow.keras import Model, Input # type: ignore
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, SpatialDropout1D, Bidirectional # type: ignore

def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)

def convert_text_to_sequences(text):
    text_sequences = tokenizer_input.texts_to_sequences(text)
    text_sequences = pad_sequences(text_sequences, maxlen=411, padding='post', value=0)
    return text_sequences


with open('.\\BiLSTM\\Tokenizer\\tokenizer_input.pkl', 'rb') as handle:
    tokenizer_input = pickle.load(handle)
with open('.\\BiLSTM\\Tokenizer\\tokenizer_target.pkl', 'rb') as handle:
    tokenizer_target = pickle.load(handle)
idx2word = pickle.load(open('.\\BiLSTM\\Dictionary\\idx2word.pkl', 'rb'))



EMBEDDING_DIM = 200
# create embedding layer
embedding_layer = Embedding(
    22220,
    EMBEDDING_DIM,
    input_length=411,
    trainable=True
)

input = Input(shape=(411,))
x = embedding_layer(input)
model = SpatialDropout1D(0.1)(x)
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(30, activation="softmax"))(model)
model = Model(input, out)

model.load_weights('.\\BiLSTM\\BiLSTM_model_final.h5')
model.summary()

def predict(input_texts:list):
    sequence = convert_text_to_sequences(input_texts)
    p = np.argmax(model.predict(sequence), axis=-1)
    output_texts = []
    for i in range(len(input_texts)):
        predict_output = ""
        user_query_split = remove_punctuation(input_texts[i])
        user_query_split = user_query_split.lower().split(" ")
        for t in range(len(user_query_split)):
            try:
                if idx2word[user_query_split[t]][p[i][t]] == 'number':
                    predict_output += '<number> '
                elif idx2word[user_query_split[t]][p[i][t]] == 'date':
                    predict_output += '<date> '
                else:
                    predict_output += str(idx2word[user_query_split[t]][p[i][t]]) + " "
            except KeyError:
                predict_output += user_query_split[t] + " "
        output_texts.append(predict_output[:-1])
    return output_texts

def accuracy_test():
    test = open('100Tests.txt', 'r').read().splitlines()
    real = open('100Tests (1).txt', 'r').read().splitlines()
    from InputHandler import InputHandler
    a = InputHandler()
    test = a.remover(test)
    test = [' '.join(sent) for sent in test]
    prediction = predict(test)
    from utility import get_accuracy
    print(get_accuracy(prediction, real))


if __name__ == '__main__':
    accuracy_test()