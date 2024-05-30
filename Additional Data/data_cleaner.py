input = "D:\\GIÁO TRÌNH 20232\\IT3190E - Machine Learning\\ML Project\\Data Collector\\Data Processing\\input.txt"
output = "D:\\GIÁO TRÌNH 20232\\IT3190E - Machine Learning\\ML Project\\Data Collector\\Data Processing\\output.txt"
final = "D:\\GIÁO TRÌNH 20232\\IT3190E - Machine Learning\\ML Project\\Data Collector\\Data Processing\\final.txt"

import re 

def date_cleaner (input, output):
    text = open(input, "r", encoding="utf-8")
    out = open(output, "a", encoding="utf-8")
    text_data = [line.strip("\n") for line in text.readlines()]

    pattern1 = r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}"
    pattern2 = r"Ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}"
    pattern3 = r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s"
    pattern4 = r"Ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s"


    patterns = [pattern1, pattern2, pattern3, pattern4]
    replacement_text = " <date> "

    for line in text_data:
        sentence = line
        for pattern in patterns:
            sentence = re.sub(pattern, replacement_text, sentence)
        out.write(sentence + "\n")
    return output

def number_cleaner (output, final):
    text = open(output, "r", encoding="utf-8")
    out = open(final, "a", encoding="utf-8")
    text_data = [line.strip("\n") for line in text.readlines()]
    pattern1 = r"[A-Za-z0-9]{1,}/[A-Za-z0-9]{1,}/[A-ZĐa-z0-9-]{1,}" #remove 98/2010/ND-CP22 or stuff
    pattern2 = r"[A-Za-z0-9]{1,}/[A-ZĐa-z0-9 -]{1,} " #remove 98/2010/QH33 or stuff
    pattern3 = r"\d{1,}[.,]\d{1,}" #remove numbers with decimal places
    pattern4 = r"\d{2,}" #remove whole numbers with more than or equals 2 digits
    pattern5 = r"\d{1,}/\d{1,}" #remove fractions
    pattern6 = r"[\d{1,}]"

    patterns = [pattern1, pattern2, pattern3, pattern4, pattern5]

    replacement_text = " <number> "

    for line in text_data:
        sentence = line
        for pattern in patterns:
            sentence = re.sub(pattern, replacement_text, sentence)
        sentence = re.sub(pattern6, "", sentence)
        sentence = sentence.replace("  ", " ").replace(" ,", ",").replace("[]", "").replace("/TT-", "")
        out.write(sentence + "\n")
    print("Cleaning completed.")

number_cleaner(date_cleaner(input, output), final)
