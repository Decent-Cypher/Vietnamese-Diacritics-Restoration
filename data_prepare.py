import re
from tqdm import tqdm
import pickle
import csv
from InputHandler import InputHandler
from nltk import word_tokenize
import random
from sklearn.model_selection import train_test_split

dataset_file = "corpus-full/corpus-full-v2.txt"

def remove_vn_accent(text):
    '''
        - Remove all accents from a vietnamese word and returned the diacritics-less version of it
        - The input is required to be in lowercase
    '''
    text = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', text)
    text = re.sub('[éèẻẽẹêếềểễệ]', 'e', text)
    text = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', text)
    text = re.sub('[íìỉĩị]', 'i', text)
    text = re.sub('[úùủũụưứừửữự]', 'u', text)
    text = re.sub('[ýỳỷỹỵ]', 'y', text)
    text = re.sub('đ', 'd', text)
    return text

def remove_diacritics(utf8_str):
    intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
    intab_u = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    intab = list(intab_l+intab_u)
    outtab_l = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"
    outtab_u = "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"
    outtab = outtab_l + outtab_u
    # Khởi tạo regex tìm kiếm các vị trí nguyên âm có dấu 'ạ|ả|ã|...'
    r = re.compile("|".join(intab))
    # Dictionary có key-value là từ có dấu-từ không dấu. VD: {'â' : 'a'}
    replaces_dict = dict(zip(intab, outtab))
    # Thay thế các từ có dấu xuất hiện trong tìm kiếm của regex bằng từ không dấu tương ứng
    non_dia_str = r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)
    return non_dia_str

def gen_accents_word(word):
    '''
        - Generate a list of all possible versions with diacritics of a word and return it
    '''
    word_no_accent = remove_vn_accent(word.lower())
    all_accent_word = {word}
    for w in open('others/all-vietnamese-syllables.txt', encoding='utf-8').read().splitlines():
        w_no_accent = remove_vn_accent(w.lower())
        if w_no_accent == word_no_accent:
            all_accent_word.add(w)
    return all_accent_word

def create_data_small():
    data_file_obj = open(dataset_file, 'r')
    raw_data = data_file_obj.read(80000000).split('\n')
    print(len(raw_data))
    train_X_500k = []
    train_Y_500k = []
    val_X_50k = []
    val_Y_50k = []
    test_X_50k = []
    test_Y_50k = []

    for i in tqdm(range(600000)):
        origin_seq = raw_data[i]
        try:
            non_acc_seq = remove_diacritics(origin_seq)
        except:
            print('error remove diacritics at sequence {}', str(i))
            next
        if i < 500000:
            train_Y_500k.append(origin_seq)
            train_X_500k.append(non_acc_seq)
        elif i < 550000:
            val_Y_50k.append(origin_seq)
            val_X_50k.append(non_acc_seq)
        else:
            test_Y_50k.append(origin_seq)
            test_X_50k.append(non_acc_seq)
    with open('data_small/train_X_500k.pkl', 'wb+') as f:
        pickle.dump(train_X_500k, f)
    with open('data_small/train_Y_500k.pkl', 'wb+') as f:
        pickle.dump(train_Y_500k, f)
    with open('data_small/val_X_50k.pkl', 'wb+') as f:
        pickle.dump(val_X_50k, f)
    with open('data_small/val_Y_50k.pkl', 'wb+') as f:
        pickle.dump(val_Y_50k, f)
    with open('data_small/test_X_50k.pkl', 'wb+') as f:
        pickle.dump(test_X_50k, f)
    with open('data_small/test_Y_50k.pkl', 'wb+') as f:
        pickle.dump(test_Y_50k, f)
        
    data_file_obj.close()
    print(train_X_500k[20])
    print(train_Y_500k[20])
    print(val_X_50k[20])
    print(val_Y_50k[20])
    print(test_X_50k[20])
    print(test_Y_50k[20])

def create_data_large():
    data_file_obj = open(dataset_file, 'r')
    raw_data = data_file_obj.read(400000000).split('\n')
    print(len(raw_data))
    train_X_2500k = []
    train_Y_2500k = []
    val_X_250k = []
    val_Y_250k = []
    test_X_250k = []
    test_Y_250k = []

    for i in tqdm(range(3000000)):
        origin_seq = raw_data[i]
        try:
            non_acc_seq = remove_diacritics(origin_seq)
        except:
            print('error remove diacritics at sequence {}', str(i))
            next
        if i < 2500000:
            train_Y_2500k.append(origin_seq)
            train_X_2500k.append(non_acc_seq)
        elif i < 2750000:
            val_Y_250k.append(origin_seq)
            val_X_250k.append(non_acc_seq)
        else:
            test_Y_250k.append(origin_seq)
            test_X_250k.append(non_acc_seq)
    with open('data_large/train_X_2500k.pkl', 'wb+') as f:
        pickle.dump(train_X_2500k, f)
    with open('data_large/train_Y_2500k.pkl', 'wb+') as f:
        pickle.dump(train_Y_2500k, f)
    with open('data_large/val_X_250k.pkl', 'wb+') as f:
        pickle.dump(val_X_250k, f)
    with open('data_large/val_Y_250k.pkl', 'wb+') as f:
        pickle.dump(val_Y_250k, f)
    with open('data_large/test_X_250k.pkl', 'wb+') as f:
        pickle.dump(test_X_250k, f)
    with open('data_large/test_Y_250k.pkl', 'wb+') as f:
        pickle.dump(test_Y_250k, f)
        
    data_file_obj.close()
    print(train_X_2500k[120])
    print(train_Y_2500k[120])
    print(val_X_250k[120])
    print(val_Y_250k[120])
    print(test_X_250k[120])
    print(test_Y_250k[120])



def save_testset(X_test_filename = 'testset_200\\test_X_200.pkl', Y_test_filename = 'testset_200\\test_Y_200.pkl', size = 200, max_length = 7, min_length = 4):
    X_test = pickle.load(open('data_small\\test_X_50k.pkl', 'rb'))
    Y_test = pickle.load(open('data_small\\test_Y_50k.pkl', 'rb'))
    test_X_new = []
    test_Y_new = []
    count = 0
    for i in tqdm(range(50000)):
        if count == size:
            break
        X_sequence = X_test[i]
        Y_sequence = Y_test[i]
        seq_len = len(X_sequence.split()) 
        if seq_len <= max_length and seq_len >= min_length:
            print(Y_sequence) # Debugging line
            test_X_new.append(X_sequence)
            test_Y_new.append(Y_sequence)
            count += 1
    with open(X_test_filename, 'wb+') as f:
        pickle.dump(test_X_new, f)
    with open(Y_test_filename, 'wb+') as f:
        pickle.dump(test_Y_new, f)

def save_testset_CSV(X_test_filename = 'testset_200\\test_X_200_1.csv', Y_test_filename = 'testset_200\\test_Y_200_1.csv', size = 200, max_length = 7, min_length = 4):
    X_test = pickle.load(open('data_small\\test_X_50k.pkl', 'rb'))
    Y_test = pickle.load(open('data_small\\test_Y_50k.pkl', 'rb'))
    test_X_new = []
    test_Y_new = []
    count = 0
    for i in tqdm(range(40000, 50000)):
        if count == size:
            break
        X_sequence = X_test[i]
        Y_sequence = Y_test[i]
        seq_len = len(X_sequence.split()) 
        if seq_len <= max_length and seq_len >= min_length:
            print(Y_sequence) # Debugging line
            test_X_new.append(X_sequence)
            test_Y_new.append(Y_sequence)
            count += 1

    # Open the CSV file in append mode
    with open(X_test_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the new single line to the CSV file
        for sequence in test_X_new:
            writer.writerow([sequence])

    # Open the CSV file in append mode
    with open(Y_test_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the new single line to the CSV file
        for sequence in test_Y_new:
            writer.writerow([sequence])

def import_CSV_to_pkl(X_test_csv = 'testset_200\\test_X_200_1.csv', Y_test_csv = 'testset_200\\test_Y_200_1.csv', X_test_pkl = 'testset_200\\test_X_200.pkl', Y_test_pkl = 'testset_200\\test_Y_200.pkl'):
    X_test = []
    Y_test = []
    with open(X_test_csv, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
        for row in rows:
            X_test.append(row[0])
    with open(Y_test_csv, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
        for row in rows:
            Y_test.append(row[0])
            print(row[0])
    with open(X_test_pkl, 'wb+') as f:
        pickle.dump(X_test, f)
    with open(Y_test_pkl, 'wb+') as f:
        pickle.dump(Y_test, f)

def change_syllables(data_filename):
    data = pickle.load(open(data_filename, 'rb'))
    syllable_pairs = [['ỏe', 'oẻ'], ['óe', 'oé'], ['òe', 'oè'], ['õe', 'oẽ'], ['ọe', 'oẹ'],
                      ['õa', 'oã'], ['óa', 'oá'], ['òa', 'oà'], ['ỏa', 'oả'], ['ọa', 'oạ'],
                      ['úy', 'uý'], ['ùy', 'uỳ'], ['ủy', 'uỷ'], ['ũy', 'uỹ'], ['ụy', 'uỵ']]
    count = 0
    for j in range(len(data)):
        p = InputHandler()
        try:
            tokens = p.remover(data[j])
        except IndexError:
            print('ERROR here')
            print(data[j])

            break
        
        continue
        # print(tokens)
        changed = False
        for i in range(len(tokens)):
            for s in syllable_pairs:
                if tokens[i][-2:] == s[0]:
                    tokens[i] = re.sub(s[0], s[1], tokens[i])
                    changed = True
        
        data[j] = p.converter(tokens)
        if changed:
            
            print(data[j])
            count += 1
        if not changed:
            print("NO")
        print()
        #     print('-> '+data[j])

    

if __name__ == '__main__':
    # data = pickle.load(open('final.pkl', 'rb'))
    # Y = data
    # random.shuffle(Y)
    # X = []
    # for sent in tqdm(Y):
    #     X.append(remove_diacritics(sent))
    # X_train, X_test, y_train, y_test = train_test_split(X, Y , random_state=104,test_size=0.25, shuffle=True)
    # with open('Additional Data\\final_train_X.pkl', 'wb+') as f:
    #     pickle.dump(X_train, f)
    # with open('Additional Data\\final_train_Y.pkl', 'wb+') as f:
    #     pickle.dump(y_train, f)
    # with open('Additional Data\\final_test_X.pkl', 'wb+') as f:
    #     pickle.dump(X_test, f)
    # with open('Additional Data\\final_test_Y.pkl', 'wb+') as f:
    #     pickle.dump(y_test, f)

    i = 5
    data = pickle.load(open('Additional Data\\final_test_X.pkl', 'rb'))
    print(data[i])
    data = pickle.load(open('Additional Data\\final_test_Y.pkl', 'rb'))
    print(data[i])


    
    

