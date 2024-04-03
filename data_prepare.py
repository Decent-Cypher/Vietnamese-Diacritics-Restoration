import re
from tqdm import tqdm
import pickle

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
    for w in open('others/all-vietnamese-syllables.txt').read().splitlines():
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


if __name__ == '__main__':
    create_data_large()

