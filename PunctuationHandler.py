import re
import string

# - Xin chao, toi la B.

# punctuation: "-   xin   ,        ."
# capital: vị trí trong câu gốc có capitalization
# token_position: vị trí bắt đầu mỗi token trong câu gốc

# ['xin', 'chao', 'toi', 'la', 'b']

class PunctuationHandler():
    def __init__(self):
        # self.texts = []
        self.special_characters = []
        self.capital = []
        self.token_position = []
        # self.tokenized_output = []

    def remover(self, texts : list[str]):
        tokenized_output = []
        token_pattern = r"[A-Za-z0-9ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]+([^ ]*[A-Za-z0-9ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]+)?"
        special_char_pattern = r"([!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\\ ]*[ ]+[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\\ ]*|^[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\\]+|[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\\]+$)"
        capital_pattern = r"[A-ZẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]"
        # self.texts = texts
        for text in texts:
            token_matches = re.finditer(token_pattern, text)
            sent_tokens = []
            sent_tokens_pos = []
            sent_special_char = []
            sent_capital = []
            for match in token_matches:
                matched_text = match.group()
                start_pos = match.start()
                end_pos = match.end()
                sent_tokens_pos.append([matched_text, start_pos])
                sent_tokens.append(matched_text.lower())
                # print(f"Token match found from {start_pos} to {end_pos}")
            special_matches = re.finditer(special_char_pattern, text)
            for match in special_matches:
                matched_text = match.group()
                sent_special_char.append([matched_text, match.start()])
                print(f"Special match found: {matched_text}, start at {match.start()}")
            capital_matches = re.finditer(capital_pattern, text)
            for match in capital_matches:
                # matched_text = match.group()
                sent_capital.append(match.start())
                # print(f"Capital match found: {matched_text}, start at {match.start()}")
            tokenized_output.append(sent_tokens)
            self.token_position.append(sent_tokens_pos)
            self.special_characters.append(sent_special_char)
            self.capital.append(sent_capital)
        return tokenized_output
    
    def converter(self, input: list[list[str]]):
        output = []
        for i in range(len(input)):
            sent_tokens = input[i]
            assert len(sent_tokens) == len(self.token_position[i])
            sent = ""
            idx = 0
            start_char_l = list(filter(lambda x: x[1] == 0, self.special_characters[i]))
            for c in start_char_l:
                sent += c[0]
                idx += len(c[0])
            for j in range((len(sent_tokens))):
                token = sent_tokens[j]
                assert len(token) == len(self.token_position[i][j][0]) and idx == self.token_position[i][j][1]
                sent += token
                for c in self.capital[i]:
                    if c >= idx and c < idx + len(token):
                        sent = sent[:c] + sent[c].upper() + sent[c+1:]
                idx = idx + len(token)
                special_char_l = list(filter(lambda x: x[1] == idx, self.special_characters[i]))
                for c in special_char_l:
                    sent += c[0]
                    idx += len(c[0])
            output.append(sent)
        return output

if __name__ == "__main__":
    p = PunctuationHandler()
    a = p.remover([" - Hôm, nay !là   Char%Les%29.!  \" HuHufnjnR@ \]\""])
    print(a)
    print(p.converter(a)[0])

