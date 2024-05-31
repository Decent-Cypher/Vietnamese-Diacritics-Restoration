import json
import pickle
import re
import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from InputHandler import InputHandler


#############  SERIALIZE KERAS OBJECTS  #############
def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


@keras.saving.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


@keras.saving.register_keras_serializable()
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-7)
        self.add = tf.keras.layers.Add()


# Global attention layer
@keras.saving.register_keras_serializable()
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


@keras.saving.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-7)

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


@keras.saving.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


@keras.saving.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


#############  DESERIALIZE KERAS OBJECTS  #############


# Step 1: Load The model
# load json and create model
json_file = open('Transformer\\Model\\transformer_final.json', 'r')
model_json = json_file.read()
json_file.close()

model = keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("Transformer\\Model\\transformer_weights_final.h5")
print("Loaded model from disk")

# Step 2: Convert Text To Sequences
filename = 'Transformer\\Dictionary\\tokenizer_final.pkl'
tokenizer = pickle.load(open(filename, 'rb'))
# print(tokenizer.texts_to_sequences(["Hom nay troi dep, troi trong xanh bao la"]))

# Step 3: Load word-index json file
with open('Transformer\\Dictionary\\word2idx_final.json', 'r') as file:
    # Load JSON data into a dictionary
    word2idx = json.load(file)

with open('Transformer\\Dictionary\\idx2word_final.json', 'r') as file:
    idx2word = json.load(file)

# Step 4: Create a function convert sentences to sequences
MAX_LEN_INPUT = 199


def convert_text_to_sequences(text: list):
    text_sequences = tokenizer.texts_to_sequences(text)
    text_sequences = pad_sequences(text_sequences, maxlen=MAX_LEN_INPUT, padding='post', value=0)
    return text_sequences


# Step 5: Create a function to remove punctuation from the sentence
def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)


# Step 6: Create a function to get the prediction from the model
def get_prediction(list_of_queries: list):
    punctual_handler = InputHandler()
    punctual_handler.remover(list_of_queries)
    text_queries = convert_text_to_sequences(list_of_queries)
    p = model.predict(text_queries)
    p = np.argmax(p, axis=-1)
    output_list_temp = []
    for query in list_of_queries:
        user_query_split = remove_punctuation(query).lower().split(" ")
        predict_output_temp = []
        for t in range(len(user_query_split)):
            if user_query_split[t] not in idx2word.keys() or str(p[0][t]) not in idx2word[user_query_split[t]].keys():
                predict_output_temp.append(user_query_split[t])
            else:
                predict_output_temp.append(str(idx2word[user_query_split[t]][str(p[0][t])]))
        output_list_temp.append(predict_output_temp)
    predict_output = punctual_handler.converter(output_list_temp)
    return predict_output


# Step 7: Test the result
if __name__ == "__main__":
    test = open('Test_100_X_long.txt', 'r').read().splitlines()
    real = open('Test_100_Y_long.txt', 'r').read().splitlines()
    from InputHandler import InputHandler
    a = InputHandler()
    test = a.remover(test)

    b = InputHandler()
    test = [' '.join(sent) for sent in test]
    print('--------------------------------------')
    prediction = get_prediction(test)
    from utility import get_accuracy
    print(get_accuracy(prediction, real))

