#coding:utf-8
#nltk.tokenize 是NLTK提供的分词工具包。所谓的分词 (tokenize) 实际就是把段落分成句子，把句子分成一个个单词的过程。我们导入的 sent_tokenize() 函数对应的是分段为句。 word_tokenize()函数对应的是分句为词。
from nltk.tokenize import sent_tokenize, word_tokenize
#stopwords 是一个列表，包含了英文中那些频繁出现的词，如am, is, are。
from nltk.corpus import stopwords
#defaultdict 是一个带有默认值的字典容器。
from collections import defaultdict
#punctuation 是一个列表，包含了英文中的标点和符号。
from string import punctuation
#nlargest() 函数可以很快地求出一个容器中最大的n个数字。
from heapq import nlargest
import math
from itertools import product, count

#stopwords包含的是我们在日常生活中会遇到的出现频率很高的词，如do, I, am, is, are等等，这种词汇是不应该算是我们的关键字。同样的标点符号（punctuation）也不能被算作是关键字。
stopwords = set(stopwords.words('english') + list(punctuation))

"""
传入两个句子
返回这两个句子的相似度
"""
def calculate_similarity(sen1, sen2):
    # 设置counter计数器
    counter = 0
    try:
        num = math.log(len(sen1)) + math.log(len(sen2))
        counter = 0
        for word in sen1:
            if word in sen2:
                counter += 1
        return counter / num
    except Exception as E:
        return 0.0



"""
传入句子列表
返回各个句子之间相似度的图（邻接矩阵表示）
"""
def create_graph(word_sent):
    num = len(word_sent)
    # 初始化表: n行n列，值为0的二维数组
    board = [[0.0 for _ in range(num)] for _ in range(num)]

    for i, j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = calculate_similarity(word_sent[i], word_sent[j])
    return board


"""
输入相似度邻接矩阵
返回各个句子的分数
"""
def weighted_pagerank(weight_graph):
    # 把初始的分数值设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]

    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]

        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores


"""
判断前后分数有没有变化
这里认为前后差距小于0.0001，分数就趋于稳定
"""
def different(scores, old_scores):
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
            flag = True
            break
    return flag


"""
根据公式求出指定句子的分数
"""
def calculate_score(weight_graph, scores, i):
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0


    for j in range(length):
        fraction = 0.0
        denominator = 0.0
        # 先计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
        added_score += fraction / (denominator+1)
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score

    return weighted_score


"""
输入文本和想要提取的句子数n
返回一个列表，列表中包含所提取的排名前n的句子
"""
def Summarize(text):
    # 首先分出句子
    textlist = list(text)
    for i in range(len(textlist)-1):
        if (textlist[i] == '.') and (ord(textlist[i+1])<91) and (ord(textlist[i+1])>64):
            textlist.insert(i+1,' ')
    text = ''.join(textlist)
    sents = sent_tokenize(text)
    n = int(len(sents) * 0.4)
    # 然后分出单词
    # word_sent是一个二维的列表
    # word_sent[i]代表的是第i句
    # word_sent[i][j]代表的是第i句中的第j个单词

    word_sent = [word_tokenize(s.lower()) for s in sents]

    # 把停用词去除
    for i in range(len(word_sent)):
        for word in word_sent[i]:
            if word in stopwords:
                word_sent[i].remove(word)
    similarity_graph = create_graph(word_sent)
    scores = weighted_pagerank(similarity_graph)
    sent_selected = nlargest(n, zip(scores, count()))
    sent_index = []
    for i in range(n):
        sent_index.append(sent_selected[i][1])
    return [sents[i] for i in sent_index]



if __name__ == '__main__':
    #text='''I am a car. And I like a car. It is a car. Do you know a car?'''
    with open(r"C:\Users\HP\桌面\www.txt", "r",encoding='utf-8') as myfile:
        text = myfile.read().replace('\n' , '')
    # 可以更改参数 2 来获得不同长度的摘要。
    # 但是摘要句子数量不能多于文本本身的句子数。
    print(Summarize(text, 2))