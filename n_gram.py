import pickle
import re
from nltk import word_tokenize
import string
from tqdm import tqdm
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace, StupidBackoff, AbsoluteDiscountingInterpolated, KneserNeyInterpolated, WittenBellInterpolated, Vocabulary
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utility import accuracy_score
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from PunctuationHandler import PunctuationHandler
from data_prepare import gen_accents_word

def tokenize(sent: str):
    """
    This function returns a list of tokens from an input string.
    """
    # Convert input into lowercase and split them into tokens of words and punctuations
    tokens = word_tokenize(sent.lower())
    # Remove all punctuations
    table = str.maketrans('', '', string.punctuation) 
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens

def preprocess_corpus(train_filename, min_frequency = 4):
    Y_train = pickle.load(open(train_filename, "rb"))
    corpus = [tokenize(sent) for sent in Y_train]
    count = defaultdict(lambda: 0)
    for sent in tqdm(corpus):
        for token in sent:
            count[token] += 1
    rare_words = []
    for k, v in count.items():
        if v < min_frequency:
            rare_words.append(k)
    for sent in tqdm(corpus):
        for i in range(len(sent)):
            if sent[i] in rare_words:
                sent[i] = "<UNK>"
    return corpus

def preprocess_train(train_filename='Y_train_new.pkl', n=3, unk_cutoff=4):
    # Y_train = pickle.load(open(train_filename, "rb"))
    with open(train_filename, 'r', encoding='utf-8') as f:
        Y_train = f.readlines()
    p = PunctuationHandler()
    corpus = p.remover(Y_train)
    train_data, padded_sents = padded_everygram_pipeline(n, corpus)
    vocab = Vocabulary(padded_sents, unk_cutoff=unk_cutoff)
    print("Finished preprocessing training data.")
    return train_data, padded_sents, vocab

def save_model(train_data, padded_sents, model, n = 3, model_filename = 'kneserney_ngram.pkl'):
    # Create a corpus of words from training set
    # corpus = preprocess_corpus(train_filename)

    # train_data (list[list[tuple]]): a list whose element is a list of all subsets of all padded trigrams in a sentence.
    # padded_sents (list): list of all tokens of all padded sentences
    # train_data, padded_sents = padded_everygram_pipeline(n, corpus)
    
    model.fit(train_data, padded_sents)

    with open(model_filename, 'wb') as fout:
        pickle.dump(model, fout)

    print("Saved model.")

def generate_sent(model, num_words, pre_words=[]):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    detokenize = TreebankWordDetokenizer().detokenize
    content = pre_words
    for i in range(num_words):
        token = model.generate(1, text_seed=content[-2:])
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)

# beam search
# def beam_search(words, model, k=3):
#   """
#     words: A list of words representing the input sequence.
#     model: A language model object used to score the likelihood of word sequences. 
#     k: Beam width, i.e., the number of sequences to keep at each step. Default value is 3.
#   """
#   sequences = []
#   for idx, word in enumerate(words):
#     if idx == 0:
#       sequences = [([x], 0.0) for x in gen_accents_word(word)]
#     else:
#       all_sequences = []
#       for seq in sequences:
#         for next_word in gen_accents_word(word):
#           current_word = seq[0][-1]
#           try:
#               previous_word = seq[0][-2]
#               score = model.logscore(next_word, [previous_word, current_word])
#           except:
#               score = model.logscore(next_word, [current_word])
#           new_seq = seq[0].copy()
#           new_seq.append(next_word)
#           all_sequences.append((new_seq, seq[1] + score))
#       all_sequences = sorted(all_sequences,key=lambda x: x[1], reverse=True)
#       sequences = all_sequences[:k]
#   return sequences

def beam_search(tokens, model, b=3):
    """
    tokens: A list of words representing the input sequence.
    model: A language model object used to score the likelihood of word sequences. 
    b: Beam width, i.e., the number of sequences to keep at each step. Default value is 3.
    """
    candidates = []
    l = len(tokens)
    for idx in range(l+1):
        d = [] # hold the probability of every combination of words
        if idx == 0:
            accent_words = gen_accents_word(tokens[idx])
            for word in accent_words:
                log_score = model.logscore(word, ['<s>', '<s>'])
                d.append([[word], log_score])
                # print(f"{word} {log_score}")
        elif idx == l:
            for candidate in candidates:
                log_score = model.logscore('</s>', [candidate[0][idx-2], candidate[0][idx-1]])
                d.append([candidate[0], log_score + candidate[1]])
                # print(f'Probability of "EOS" given {candidate[0]}: {log_score}')
                # print(f'Probability of "{candidate[0]}: {log_score + candidate[1]}')
        else:
            accent_words = gen_accents_word(tokens[idx])
            for candidate in candidates:
                for word in accent_words:
                    if idx == 1:
                        log_score = model.logscore(word, ['<s>', candidate[0][idx-1]])
                    else:
                        log_score = model.logscore(word, [candidate[0][idx-2], candidate[0][idx-1]])
                    # print(f'Probability of "{word}" given {candidate[0]}: {log_score}')
                    # print(f'Probability of "{candidate[0]+[word]}: {log_score + candidate[1]}')
                    d.append([candidate[0]+[word], log_score + candidate[1]])
        # print(f"{idx}. \n{d}")
        candidates = sorted(d, key=lambda x: x[1], reverse=True)[:b]
        # print(candidates)
    return candidates


def predict(model_filename, texts: list[str]):
    output = []
    p = PunctuationHandler()
    tokenized_texts = p.remover(texts)
    model = pickle.load(open(model_filename, "rb"))
    for sentence in tokenized_texts:
        result = beam_search(sentence, model)
        # print("After accents insertion: " + p.converter(result[0][0]))
        output.append(result[0][0])
    return p.converter(output)

def predict_testset(model_filename = "kneserney_ngram.pkl", X_test_filename = 'testset_200\\test_X_200.pkl', Y_pred_csv = 'pred_Y_200.csv'):
    # detokenize = TreebankWordDetokenizer().detokenize
    model = pickle.load(open(model_filename, "rb"))
    X_test = pickle.load(open(X_test_filename, 'rb'))
    for i in tqdm(range(len(X_test))):
        with open(Y_pred_csv, mode='w+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            sentence = X_test[i]
            input_tokens = tokenize(sentence)
            print(input_tokens)
            result = beam_search(input_tokens, model)
            sentence_pred = ' '.join(result[0][0])
            print("After accents insertion: " + sentence_pred)
            writer.writerow([sentence_pred])

def print_report(Y_pred_csv = 'pred_Y_200.csv', Y_test_pkl = 'testset_200\\test_Y_200.pkl'):
    Y_test_sequences = pickle.load(open(Y_test_pkl, 'rb'))
    X_test = pickle.load(open('testset_200\\test_X_200.pkl', 'rb'))
    Y_test = pad_sequences([sentence.lower().split() for sentence in Y_test_sequences], padding='post', dtype=object)
    Y_test_arr = np.array(Y_test)
    Y_pred = []
    with open(Y_pred_csv, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
        for i in range(len(rows)):
            seq_pred = rows[i][0]
            Y_pred.append(list(seq_pred.split()))
    Y_pred_arr = np.array(pad_sequences(Y_pred, padding='post', dtype=object))

    print('Testing Set Accuracy:', accuracy_score(Y_test_arr, Y_pred_arr))    
    
if __name__ == "__main__":
    train_filename = 'Final_Political.txt'
    n = 3
    train_data, padded_sents, vocab = preprocess_train(train_filename=train_filename, unk_cutoff=1)
    model = KneserNeyInterpolated(order=n, vocabulary=vocab)  # discount = 0.1
    # model = Laplace(order=n, vocabulary=vocab)
    # model = StupidBackoff(order=n, vocabulary=vocab) # alpha = 0.4
    model_filename = 'N_gram_model\\kneserney_trigram_political_cutoff1.pkl'
    save_model(train_data=train_data, padded_sents=padded_sents, model = model, n = n, model_filename = model_filename)
    sentence = 'Doi voi giao duc, can su dong thuan giua gia dinh, nha truong va xa hoi.'
    print(predict(model_filename=model_filename, texts=[sentence,]))





                