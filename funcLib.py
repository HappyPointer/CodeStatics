import data.cnews_loader
import xlrd


# 该函数将从特定的位置读取英文垃圾邮件分类训练集
# 注：如果你的项目文件中没有这些文件，该函数会出错
def readEnglish():
    textData = []    # 储存文本数据
    labelData = []   # 储存标签数据
    f = open("./Data/SMSSpamCollection", "r", encoding='utf-8')

    for line in f:
        array = line.split('\t')
        labelData.append(array[0])
        textData.append(array[1])
    f.close()
    return [textData, labelData]


# 该函数接受一个英文字符串，对该字符串进行特殊字符过滤、词干提取后，返回经过处理的英文文本
def processEnglish(textData):
    processedData = []
    for line in textData:
        processedSentence = data.cnews_loader.clean_str(line)
        processedData.append(processedSentence)

    return processedData




def transformVec2Id(textList, word2intList):  # 这里转化后的词向量长度为 64
    reversedList = []
    for text in textList:
        singleText = []
        for word in text:
            if word in word2intList:
                singleText.append(word2intList[word])
            else:
                singleText.append(0)

        if len(singleText) > 64:
            singleText = singleText[:64]

        while len(singleText) < 64:  # padding
            singleText.append(0)


        singleText = singleText[::-1]
        reversedList.append(singleText)
    return reversedList


def transformLabel2Onehot(labelList):
    reversedList = []
    for label in labelList:
        if label == 'P':
            reversedList.append([1, 0, 0])
        elif label == "I":
            reversedList.append([0, 1, 0])
        else:
            reversedList.append([0, 0, 1])
    return reversedList


def transformLabel2Onehot_with4classes(labelList):
    reversedList = []
    for label in labelList:
        if label == 'P':
            reversedList.append([1, 0, 0, 0])
        elif label == "I":
            reversedList.append([0, 1, 0, 0])
        elif label == "O":
            reversedList.append([0, 0, 1, 0])
        else:
            reversedList.append([0, 0, 0, 1])
    return reversedList


def readTextRank(filename):
    resultList = []
    # 打开工作表
    workbook = xlrd.open_workbook(filename=filename)
    # 用索引取第一个工作薄
    booksheet = workbook.sheet_by_index(0)
    # 返回的结果集
    for i in range(booksheet.nrows):
        resultList.append(booksheet.row_values(i)[0][6:])
    return resultList


def getQulifiedData(labeledText, candidateDataText):
    generatedTextList = []
    processCandidateData = []
    for line in candidateDataText:
        processCandidateData.append(processEnglish(line))
    for line in candidateDataText:
        if line not in labeledText:
            generatedTextList.append(line)
    return generatedTextList

def compare(a,b,c,d):
    if a>10*b and a>10*c and a>10*d:
        return True
    else:
        return False




# train_x, train_y = data.cnews_loader.read_file("./data/cnews/cnews.train.txt")
# candidate_data = readTextRank("./DataFile/test.xls")
# candidate_data = processEnglish(candidate_data)
# generatedData = getQulifiedData(train_x, candidate_data)
# print(generatedData)




