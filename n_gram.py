import pickle
import re
from nltk import word_tokenize
import string
from tqdm import tqdm
from nltk.util import bigrams, trigrams
from nltk.lm.preprocessing import pad_both_ends
from collections import Counter, defaultdict
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace, StupidBackoff,KneserNeyInterpolated, Vocabulary
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utility import accuracy_score
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from InputHandler import InputHandler
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
    Y_train = pickle.load(open(train_filename, "rb"))
    # with open(train_filename, 'r', encoding='utf-8') as f:
    #     Y_train = f.readlines()
    p = InputHandler()
    corpus = p.remover(Y_train)
    train_data, padded_sents = padded_everygram_pipeline(n, corpus)
    # for p in padded_sents:
    #     print(p)
    vocab = Vocabulary(padded_sents, unk_cutoff=unk_cutoff)
    print("Finished preprocessing training data.")
    return train_data, padded_sents, vocab

def save_model(train_data, padded_sents, model, model_filename = 'kneserney_ngram.pkl'):
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

def beam_search(tokens, model, b=3):
    """
    tokens: A list of words representing the input sequence.
    model: A language model object used to score the likelihood of word sequences. 
    b: Beam width, i.e., the number of sequences to keep at each step. Default value is 3.
    """
    n = model.order
    # print(n)
    candidates = []
    if n == 3:
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
    elif n == 2:
        l = len(tokens)
        for idx in range(l+1):
            d = [] # hold the probability of every combination of words
            if idx == 0:
                accent_words = gen_accents_word(tokens[idx])
                for word in accent_words:
                    log_score = model.logscore(word, ['<s>'])
                    d.append([[word], log_score])
                    # print(f"{word} {log_score}")
            elif idx == l:
                for candidate in candidates:
                    log_score = model.logscore('</s>', [candidate[0][idx-1]])
                    d.append([candidate[0], log_score + candidate[1]])
                    # print(f'Probability of "EOS" given {candidate[0]}: {log_score}')
                    # print(f'Probability of "{candidate[0]}: {log_score + candidate[1]}')
            else:
                accent_words = gen_accents_word(tokens[idx])
                for candidate in candidates:
                    for word in accent_words:
                        log_score = model.logscore(word, [candidate[0][idx-1]])
                        # print(f'Probability of "{word}" given {candidate[0]}: {log_score}')
                        # print(f'Probability of "{candidate[0]+[word]}: {log_score + candidate[1]}')
                        d.append([candidate[0]+[word], log_score + candidate[1]])
            # print(f"{idx}. \n{d}")
            candidates = sorted(d, key=lambda x: x[1], reverse=True)[:b]
            # print(candidates)
            # print()
    elif n == 1:
        l = len(tokens)
        for idx in range(l):
            d = [] # hold the probability of every combination of words
            accent_words = gen_accents_word(tokens[idx])
            if idx == 0:
                for word in accent_words:
                    log_score = model.logscore(word)
                    d.append([[word], log_score])
                    # print(f"{word} {log_score}")
            else:
                for candidate in candidates:
                    for word in accent_words:
                        log_score = model.logscore(word)
                        # print(f'Probability of "{word}" given {candidate[0]}: {log_score}')
                        # print(f'Probability of "{candidate[0]+[word]}: {log_score + candidate[1]}')
                        d.append([candidate[0]+[word], log_score + candidate[1]])
            # print(f"{idx}. \n{d}")
            candidates = sorted(d, key=lambda x: x[1], reverse=True)[:b]
            # print(candidates)
    return candidates


def predict(model_filename, texts: list[str], b=3):
    output = []
    p = InputHandler()
    tokenized_texts = p.remover(texts)
    model = pickle.load(open(model_filename, "rb"))
    for sentence in tokenized_texts:
        result = beam_search(sentence, model, b)
        # print("After accents insertion: " + p.converter(result[0][0]))
        output.append(result[0][0])
    return p.converter(output)

def predict_testset(model_filename = "kneserney_ngram.pkl", X_test_filename = 'test_X_100.pkl', Y_pred_txt = 'pred_Y_100.txt', b=3):
    # detokenize = TreebankWordDetokenizer().detokenize
    X_test = pickle.load(open(X_test_filename, 'rb'))
    output = []
    p = InputHandler()
    tokenized_texts = p.remover(X_test)
    model = pickle.load(open(model_filename, "rb"))
    for sentence in tqdm(tokenized_texts):
        result = beam_search(sentence, model, b)
        # print("After accents insertion: " + p.converter(result[0][0]))
        output.append(result[0][0])
        print(result[0][0])
    Y_pred = p.converter(output)
    with open(Y_pred_txt, mode='w', encoding='utf-8') as f:
        for s in Y_pred:
            f.write(s + "\n")

def write_report(model_filename, Y_pred_filename = 'pred_Y_200.txt', Y_test_filename = 'testset_200\\test_Y_200.pkl'):
  with open(Y_pred_filename, 'r', encoding='utf-8') as f:
    Y_pred = f.readlines()
  h = InputHandler()
  Y_pred_tokenized = h.remover(Y_pred)
  Y_test = pickle.load(open(Y_test_filename, 'rb'))
  h1 = InputHandler()
  Y_test_tokenized = h1.remover(Y_test)
  Y_pred_padded = pad_sequences(Y_pred_tokenized, padding='post', dtype=object)
  Y_test_padded = pad_sequences(Y_test_tokenized, padding='post', dtype=object)
  accuracy = accuracy_score(np.array(Y_test_padded), np.array(Y_pred_padded))
  print('Testing Set Accuracy:', accuracy)

#   Y_test_ngrams = []
#   model = pickle.load(open(model_filename, "rb"))
#   n = model.order
#   print(n)
#   if n == 1:
#     for sent in Y_test_tokenized:
#       Y_test_ngrams.extend(sent)
#   elif n == 2:
#     for sent in Y_test_tokenized:
#       Y_test_ngrams.extend(list(bigrams(pad_both_ends(sent, n=n))))
#   elif n == 3:
#     for sent in Y_test_tokenized:
#       Y_test_ngrams.extend(list(trigrams(pad_both_ends(sent, n=n))))
#   entropy = model.entropy(Y_test_ngrams)
#   perplexity = model.perplexity(Y_test_ngrams)
#   print(entropy)
#   print(perplexity)

  with open(Y_pred_filename[:-4]+'_report'+'.txt', 'w', encoding = 'utf-8') as f:
    f.write(f'{Y_pred_filename[:-4]} report:\n')
    f.write(f'Accuracy: {accuracy}\n')
    # f.write(f'Entropy: {entropy}\n')
    # f.write(f'Perplexity: {perplexity}\n')
    
if __name__ == "__main__":
    path = "N_gram_model\\"
    # l = [ , , , , 'kneserney_trigram_scientific_cutoff1.pkl']
    # n = 1
    # train_data, padded_sents, vocab = preprocess_train('Additional Data\\final_train_Y.pkl', n=n, unk_cutoff=1)
    # model = StupidBackoff(order=n, vocabulary=vocab)
    model_filename = 'kneserney_trigram_full_cutoff2_b3.pkl'
    # save_model(train_data, padded_sents, model, path+model_filename)
    X_test_filename = 'test_100\test_X_100.pkl'
    b = 3
    Y_test_filename = 'test_100\\test_Y_100.pkl'
    Y_pred_filename = f'{model_filename[:-4]}.txt'
    # predict_testset(model_filename=model_filename, X_test_filename=X_test_filename, Y_pred_txt=Y_pred_filename, b=b)
    write_report(model_filename=model_filename, Y_pred_filename=Y_pred_filename, Y_test_filename=Y_test_filename)
    # model_filename2 = 'laplace_trigram_full_cutoff1.pkl'
    # model1 = pickle.load(open(path+model_filename,'rb'))
    # model2 = pickle.load(open(path+model_filename2,'rb'))
    # print(model1.order)
    # print(model2.order)



    








                