import re
import string

# - Xin chao, toi la B.

# punctuation: "-   xin   ,        ."
# capital: vị trí trong câu gốc có capitalization
# token_position: vị trí bắt đầu mỗi token trong câu gốc

# ['xin', 'chao', 'toi', 'la', 'b']

class InputHandler():
    def __init__(self):
        # self.texts = []
        self.special_characters = []
        self.capital = []
        self.token_position = []
        self.numbers = []
        self.dates = []
        # self.tokenized_output = []

    def remover(self, texts : list[str]):
        tokenized_output = []
        word_pattern = r"[A-Za-z0-9ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]+([^ ]*[A-Za-z0-9ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]+)?"
        special_char_pattern = r"([!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\\ ]*[ ]+[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\\ ]*|^[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\\]+|[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\\]+$)"
        capital_pattern = r"[A-ZẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]"
        date_pattern = r"([nN]g[àa]y\s+\d{1,2}\s+th[áa]ng\s+\d{1,2}(\s+n[ăa]m\s+\d{4})?)"
        number_pattern = r"([A-Za-z0-9ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]{1,}/[A-Za-z0-9ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]{1,}(/[A-ZĐa-z0-9-ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]{1,})?|\d{1,}([.,/]\d{1,})?)"
        token_pattern = r"([A-Za-z0-9ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]+([^ ]*[A-Za-z0-9ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]+)?|<date>|<number>)"
        # self.texts = texts
        for i in range(len(texts)):
            text = texts[i]

            sent_tokens = []
            sent_tokens_pos = []
            sent_special_char = []
            sent_capital = []
            sent_dates = []
            sent_numbers = []
            sent_dates_label = []
            sent_numbers_label = []
            special_matches = re.finditer(special_char_pattern, text)
            for match in special_matches:
                matched_text = match.group()
                sent_special_char.append([matched_text, match.start()])
                # print(f"Special match found: {matched_text}, start at {match.start()}")



            date_matches = re.finditer(date_pattern, text)
            for match in date_matches:
                s = match.start()
                e = match.end()
                matched_text = match.group()
                text = text[:s] + ' '*len(matched_text) + text[e:]
                sent_dates.append([matched_text, s, e])
                sent_dates_label.append(["<date>", s, e])
                # print(len(text))
            number_matches = re.finditer(number_pattern, text)
            for match in number_matches:
                s = match.start()
                e = match.end()
                matched_text = match.group()
                text = text[:s] + ' '*len(matched_text) + text[e:]
                # print(len(text))
                sent_dates_label.append(["<number>", s, e])
                sent_numbers.append([matched_text, s, e])
            # print(text)

            capital_matches = re.finditer(capital_pattern, text)
            for match in capital_matches:
                # matched_text = match.group()
                sent_capital.append(match.start())
                # print(f"Capital match found: {matched_text}, start at {match.start()}")
                
            word_matches = re.finditer(word_pattern, text)
            for match in word_matches:
                matched_text = match.group()
                start_pos = match.start()
                end_pos = match.end()
                sent_tokens_pos.append([matched_text, start_pos, end_pos])
                # print(f"Token match found from {start_pos} to {end_pos}")
            
            sent_tokens_pos = sent_tokens_pos + sent_dates_label + sent_numbers_label
            sorted_tokens = sorted(sent_tokens_pos, key = lambda x: x[1])
            for t in sorted_tokens:
                sent_tokens.append(t[0].lower())

            tokenized_output.append(sent_tokens)
            self.token_position.append(sent_tokens_pos)
            self.special_characters.append(sent_special_char)
            self.capital.append(sent_capital)
            self.dates.append(sent_dates)
            self.numbers.append(sent_numbers)

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
            # print(sent)
            for j in range((len(sent_tokens))):
                token = sent_tokens[j]
                # assert len(token) == len(self.token_position[i][j][0]) and idx == self.token_position[i][j][1]
                if token == "<date>":
                    date_l = list(filter(lambda x: x[1] == idx, self.dates[i]))
                    for c in date_l:
                        date_str = c[0]
                        date_str = re.sub('ngay', 'ngày', date_str)
                        date_str = re.sub('Ngay', 'Ngày', date_str)
                        date_str = re.sub('thang', 'tháng', date_str)
                        date_str = re.sub('nam', 'năm', date_str)
                        sent += date_str
                        idx += len(date_str)
                        # print(f'date {c[0]}')
                elif token == "<number>":
                    number_l = list(filter(lambda x: x[1] == idx, self.numbers[i]))
                    for c in number_l:
                        sent += c[0]
                        idx += len(c[0])
                        # print(f'num {c[0]}')
                else:
                    sent += token
                    idx = idx + len(token)
                # print(sent)
                special_char_l = list(filter(lambda x: x[1] == idx, self.special_characters[i]))
                for c in special_char_l:
                    sent += c[0]
                    idx += len(c[0])
                # print(sent)
            for c in self.capital[i]:
                sent = sent[:c] + sent[c].upper() + sent[c+1:]
            output.append(sent)
        return output

if __name__ == "__main__":
    i = InputHandler()
    s = "\"Harry, chúng mình cần phải\""
    out = i.remover([s,])
    print(out)
    print(i.converter(out))

