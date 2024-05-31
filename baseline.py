import unicodedata
import re
import pickle
import numpy as np
import random

target_texts = pickle.load(open('final.pkl', 'rb'))


def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)


REMOVE_DIACRITIC_TABLE = str.maketrans(
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴáàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
    "A" * 17 + "D" + "E" * 11 + "I" * 5 + "O" * 17 + "U" * 11 + "Y" * 5 + "a" * 17 + "d" + "e" * 11 + "i" * 5 + "o" * 17 + "u" * 11 + "y" * 5
)




def remove_diacritic(txt: str) -> str:
    if not unicodedata.is_normalized("NFC", txt):
        txt = unicodedata.normalize("NFC", txt)
    return txt.translate(REMOVE_DIACRITIC_TABLE)

input_texts = [remove_diacritic(sentence) for sentence in target_texts]
# Lower the text from input and output
input_texts = [txt.lower() for txt in input_texts]
target_texts = [txt.lower() for txt in target_texts]
# Create a mapping from targets to target labels

target_texts = [remove_punctuation(txt) for txt in target_texts]
word2freq = {}
for text in target_texts:
    data = text.split(' ')
    for word in data:
        remove_diacritic_word = remove_diacritic(word)
        if remove_diacritic_word not in word2freq:
            word2freq[remove_diacritic_word] = {word: 1}
        else:
            if word not in word2freq[remove_diacritic_word]:
                word2freq[remove_diacritic_word][word] = 1
            else:
                word2freq[remove_diacritic_word][word] += 1
for key in word2freq:
    s = sum([item[1] for item in word2freq[key].items()])
    word2freq[key] = {item[0]:(item[1]/s) for item in word2freq[key].items()}


def predict(texts:list):
    texts = [remove_punctuation(txt) for txt in texts]
    outs = []
    for txt in texts:
        out = ""
        for token in txt.split():
            if token not in word2freq:
                out += token + " "
            else:
                l = [item for item in word2freq[token].items()]
                a = random.choices(l, [item[1] for item in l])
                out += a[0][0] + " "
        outs.append(out[:-1])
    return outs

p = predict(input_texts)
print('Calculating accuracy of baseline model...')

from utility import get_accuracy
print(get_accuracy(p, target_texts))
    


