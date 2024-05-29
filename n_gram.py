import pickle
import re
from nltk import word_tokenize
import string
from tqdm import tqdm
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace, KneserNeyInterpolated, WittenBellInterpolated
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utility import accuracy_score
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from PunctuationHandler import PunctuationHandler

def tokenize(sent: str):
    """
    This function returns a list of tokens from an input string.
    """
    # Convert input into lowercase and split them into tokens of words and punctuations
    tokens = word_tokenize(sent.lower())
    # Remove all punctuations except _
    table = str.maketrans('', '', string.punctuation.replace("_", "")) 
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens

def save_model():
    Y_train = pickle.load(open("data_small\\train_Y_500k.pkl", "rb"))
    # Create a corpus of words from training set
    corpus = [tokenize(sent) for sent in Y_train[:2]]

    # Create a placeholder for model. Each key is a tuple of two words. 
    # Each value is a nested dictionary of the form: {next_word : frequency}.
    model = defaultdict(lambda: defaultdict(lambda: 0))

    # Count frequency of each trigram
    for sentence in tqdm(corpus):
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1

    # Transform the counts to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count

    # train_data (list[list[tuple]]): a list whose element is a list of all subsets of all padded trigrams in a sentence.
    # padded_sents (list): list of all tokens of all padded sentences
    n = 3
    train_data, padded_sents = padded_everygram_pipeline(n, corpus)

    vi_model = KneserNeyInterpolated(n)
    vi_model.fit(train_data, padded_sents)

    with open('kneserney_ngram.pkl', 'wb') as fout:
        pickle.dump(vi_model, fout)

    print("Saved model.")

def remove_vn_accent(word):
    word = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', word)
    word = re.sub('[éèẻẽẹêếềểễệ]', 'e', word)
    word = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', word)
    word = re.sub('[íìỉĩị]', 'i', word)
    word = re.sub('[úùủũụưứừửữự]', 'u', word)
    word = re.sub('[ýỳỷỹỵ]', 'y', word)
    word = re.sub('đ', 'd', word)
    return word

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

def gen_accents_word(word):
    """
    :param word: A lower-case string with no punctuations and accents
    """
    word_no_accent = word.lower()
    all_accent_word = {word}
    for w in open('all-vietnamese-syllables.txt', encoding="utf-8").read().splitlines():
        w_no_accent = remove_vn_accent(w.lower())
        if w_no_accent == word_no_accent:
            all_accent_word.add(w)
    return all_accent_word

# beam search
def beam_search(words, model, k=3):
  """
    words: A list of words representing the input sequence.
    model: A language model object used to score the likelihood of word sequences. 
    k: Beam width, i.e., the number of sequences to keep at each step. Default value is 3.
  """
  sequences = []
  for idx, word in enumerate(words):
    if idx == 0:
      sequences = [([x], 0.0) for x in gen_accents_word(word)]
    else:
      all_sequences = []
      for seq in sequences:
        for next_word in gen_accents_word(word):
          current_word = seq[0][-1]
          try:
              previous_word = seq[0][-2]
              score = model.logscore(next_word, [previous_word, current_word])
          except:
              score = model.logscore(next_word, [current_word])
          new_seq = seq[0].copy()
          new_seq.append(next_word)
          all_sequences.append((new_seq, seq[1] + score))
      all_sequences = sorted(all_sequences,key=lambda x: x[1], reverse=True)
      sequences = all_sequences[:k]
  return sequences

def beam_search1(tokens, model, b = 3):
    candidates = []
    for idx in range(len(tokens)):
        d = [] # hold the probability of every combination of words
        accent_words = gen_accents_word(tokens[idx])
        if idx == 0:
            for word in accent_words:
                log_score = model.logscore(word, ['<s>', '<s>'])
                d.append([[word], log_score])
                print(f"{word} {log_score}")
        else:
            for candidate in candidates:
                for word in accent_words:
                    if idx == 1:
                        log_score = model.logscore(word, ['<s>', candidate[0][idx-1]])
                    else:
                        log_score = model.logscore(word, candidate[0])
                    print(f'Probability of "{word}" given {candidate[0]}: {log_score}')
                    print(f'Probability of "{candidate[0]+[word]}: {log_score + candidate[1]}')
                    d.append([candidate[0]+[word], log_score + candidate[1]])
        print(f"{idx}. \n{d}")
        candidates = sorted(d, key=lambda x: x[1], reverse=True)[:3]
        print(candidates)
    return candidates


def predict_sentence(sentence: string):
    p = PunctuationHandler()
    print(p.remover(sentence))
    model = pickle.load(open("kneserney_ngram.pkl", "rb"))
    result = beam_search1(p.remover(sentence), model)
    print("After accents insertion: " + p.converter(result[0][0]))
    return p.converter(result[0][0])

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
            result = beam_search1(input_tokens, model)
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
    # predict_sentence("- Chao, toi khoe!")
    model = pickle.load(open("kneserney_ngram.pkl", "rb"))
    print(model.logscore("1"))