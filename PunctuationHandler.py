import re

# - Xin chao, toi la B.

# punctuation: "-   xin   ,        ."
# capital: vị trí trong câu gốc có capitalization
# token_position: vị trí bắt đầu mỗi token trong câu gốc

# ['xin', 'chao', 'toi', 'la', 'b']

class PunctuationHandler():
    
    
    def __init__(self) -> None:
        self.capital_only_mode = False
        self.special_characters = "!@#$%^&*()-+?_=,.<>'/\""
        # current_input = []     input split by spaces. E.g. This is ("A sentence").       ->      ["This", "is", "(\"a", "sentence\")."]
        self.pre_punctuation = []   # List of special characters at the beginning of each word  ->      ['', '', '("', '']
        self.post_punctuation = []  # List of special characters at the end of each word        ->      ['', '', '', '").']
        self.capitalization = []    # List of indices of capitalized words.                     ->      [0,2]
        self.punctuated_words = dict()
    
    def remover(self, input):
        current_input = input.split()
        for i in range(len(current_input)):
            current_word = current_input[i]
            # Check if the word is full of special characters
            if re.search("[A-Za-z0-9ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]",current_word) == None:
                self.punctuated_words[i] = current_word
                current_input[i] = ''
                self.pre_punctuation.append('')
                self.post_punctuation.append('')
                continue
            
            # Handling pre-punctuation
            if len(current_word) > 1 and current_word[1] in self.special_characters:
                self.pre_punctuation.append(current_word[0:2])
                current_word = current_word[2:]
                current_input[i] = current_word
            elif current_word[0] in self.special_characters:
                self.pre_punctuation.append(current_word[0])
                current_word = current_word[1:]
                current_input[i] = current_word
            else:
                self.pre_punctuation.append('')
            
            # Handling post-punctuation
            if len(current_word) > 3 and current_word[-3] in self.special_characters:
                self.post_punctuation.append(current_word[-3:])
                current_word = current_word[:-3]
                current_input[i] = current_word
            elif len(current_word) > 2 and current_word[-2] in self.special_characters:
                self.post_punctuation.append(current_word[-2:])
                current_word = current_word[:-2]
                current_input[i] = current_word
            elif len(current_word) > 1 and current_word[-1] in self.special_characters:
                self.post_punctuation.append(current_word[-1:])
                current_word = current_word[:-1]
                current_input[i] = current_word
            else:
                self.post_punctuation.append('')       
                
            # Handling self.capitalization
            if current_word != "" and current_word[0].isupper():
                self.capitalization.append(i)
                current_input[i] = current_word.lower()
        
        while '' in current_input:
            current_input.remove('')
            
        if not self.capital_only_mode:
            return current_input
        else:
            return input.lower().split()
            
    def converter(self, input):
        input = ' '.join(input)
        input = ''.join(e for e in input if e.isalnum() or e == ' ')
        current_input = input.split()
        for i in self.punctuated_words:
            current_input.insert(i, self.punctuated_words[i])
        
        for i in range(len(current_input)):        
            # Return self.capitalization
            if i in self.capitalization:
                current_input[i] = current_input[i].capitalize()
                
            # Return pre-punctuation
            current_input[i] = self.pre_punctuation[i] + current_input[i]
    
            # Return post-punctuation
            current_input[i] = current_input[i] + self.post_punctuation[i]
            
        output = ' '.join(current_input)
        return output

if __name__ == '__main__':
    
    sentence = 'Vay ma em ko nghi lai co ngay hom nay dau anh a, em ko muon chung ta xu co duoc ko anh"'
    a = PunctuationHandler()

    tokens = a.remover(sentence)
    print(a.self.capitalization)
    print(a.self.punctuated_words)

    print(tokens)
    
    