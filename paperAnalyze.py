# # coding=utf-8
from sys import argv
import TextRank4ZH.textrank4zh.pdfmine_copy as pdfMiner
from keyMatch import keyMatch
import textrank
import predict
import traceback
import sys
import io

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
    paper_path = argv[1]
    # paper_path = r'E:\tensorflow_code\paperAnalyze\pdf_sample\Angeli2012_22658566.pdf'

    pdf_title, pdf_content = pdfMiner.parse(paper_path)
    # print(pdf_title)
    # print(pdf_content)

    keywordsList = keyMatch(pdf_content)

    textrank_resultList = textrank.Summarize(pdf_content)
    # print(textrank_resultList)

    predictedLabelList = predict.perdictSentences(textrank_resultList)
    # print(predictedLabelList)

    P_sentences1 = list()
    I_sentences1 = list()
    O_sentences1 = list()
    U_sentences1 = list()
    for index in range(0, len(predictedLabelList)):
        label = predictedLabelList[index]
        if label[0] == "P":
            P_sentences1.append((textrank_resultList[index],label[1]))
        if label[0] == "I":
            I_sentences1.append((textrank_resultList[index],label[1]))
        if label[0] == "O":
            O_sentences1.append((textrank_resultList[index],label[1]))
        if label[0] == "U":
            U_sentences1.append((textrank_resultList[index],label[1]))

    P_sentences = sorted(P_sentences1, key=lambda tup: tup[1], reverse=True)
    I_sentences = sorted(I_sentences1, key=lambda tup: tup[1], reverse=True)
    O_sentences = sorted(O_sentences1, key=lambda tup: tup[1], reverse=True)
    U_sentences = sorted(U_sentences1, key=lambda tup: tup[1], reverse=True)

    P_sentences = P_sentences[0:5]
    I_sentences = I_sentences[0:5]
    O_sentences = O_sentences[0:5]

    # starting output the analyzed value:
    # Output title
    print(pdf_title)
    print("title_END")

    # Output P sentences
    for line in P_sentences:
        print(line[0])
    print("P_END")

    # Output I sentences
    for line in I_sentences:
        print(line[0])
    print("I_END")

    # Output O sentences
    for line in O_sentences:
        print(line[0])
    print("O_END")

    # Output Keywords
    keyword_string = ""
    if len(keywordsList) != 0:
        keyword_string = keywordsList[0]

    for index in range(1, len(keywordsList)):
        keyword_string += "&&" + keywordsList[index]
    keyword_string = keyword_string.strip()

    print(keyword_string)
    print("KEYWORD_END")


except Exception as e:
    traceback.print_exc()
    traceback.print_exc(file=open('Error_message.txt','w'))
