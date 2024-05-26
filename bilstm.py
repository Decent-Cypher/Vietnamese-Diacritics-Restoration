import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re

def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)

def convert_text_to_sequences(text):
    text_sequences = tokenizer_input.texts_to_sequences(text)
    print(text_sequences)
    text_sequences = pad_sequences(text_sequences, maxlen=200, padding='post', value=0)
    return text_sequences


with open('BiLSTM_model/tokenizer_input.pkl', 'rb') as handle:
    tokenizer_input = pickle.load(handle)
with open('BiLSTM_model/tokenizer_target.pkl', 'rb') as handle:
    tokenizer_target = pickle.load(handle)
idx2word = pickle.load(open('BiLSTM_model/idx2word.pkl', 'rb'))
model = tf.keras.models.load_model('BiLSTM_model/BiLSTM_model (8).keras')
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
                predict_output += str(idx2word[user_query_split[t]][p[i][t]]) + " "
            except KeyError:
                predict_output += user_query_split[t] + " "
        output_texts.append(predict_output[:-1])
    return output_texts

if __name__ == '__main__':
    test = [
        'Hom nay troi that la dep',
        'thay khoat cua chung em that la mot giang vien tuyet voi'
    ]
    prediction = predict(test)
    print(prediction)