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
import time

def tokenize(sent):
    tokens = word_tokenize(sent.lower())
    table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    print(tokens)
    return tokens

def save_model():
    Y_train = pickle.load(open("train_Y_500k.pkl", "rb"))
    corpus = [tokenize(sent) for sent in Y_train]

    # Create a placeholder for model
    model = defaultdict(lambda: defaultdict(lambda: 0))

    # Count frequency of co-occurance  
    for sentence in tqdm(corpus):
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1

    # Let's transform the counts to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count

    # train_data (list of tuples): all subsets of tokens of every padded trigram
    # padded_sents (list): all padded sentences
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
    detokenize = TreebankWordDetokenizer().detokenize
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
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
    word_no_accent = remove_vn_accent(word.lower())
    all_accent_word = {word}
    for w in open('all-vietnamese-syllables.txt', encoding="utf-8").read().splitlines():
        w_no_accent = remove_vn_accent(w.lower())
        if w_no_accent == word_no_accent:
            all_accent_word.add(w)
    return all_accent_word

# beam search
def beam_search(words, model, k=3):
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

def predict(sentence: string):
    detokenize = TreebankWordDetokenizer().detokenize
    model = pickle.load(open("kneserney_ngram.pkl", "rb"))
    result = beam_search(sentence.lower().split(), model)
    print("After accents insertion: " + detokenize(result[0][0]))

start_time = time.time()
sentence = "Hom nay la mot ngay dep troi"
print("Original sentence: " + sentence)
predict(sentence)
print(f"Run-time: {time.time()-start_time}")





    

