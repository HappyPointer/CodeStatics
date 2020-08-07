import tensorflow as tf
import data.cnews_loader
import funcLib
import random
import numpy as np


# 读取文本数据，并对文本进行预处理
test_x, test_y = data.cnews_loader.read_file("./data/cnews/cnews.test_withN.txt")
# 建立词向量
wordsList, word2id = data.cnews_loader.read_vocab("./data/cnews/cnews.vocab_withN.txt")
test_x = funcLib.transformVec2Id(test_x, word2id)
test_y = funcLib.transformLabel2Onehot_with4classes(test_y)


print("Test size is : " + str(len(test_x)))
# ----------------------------------------------------------------------------------------------------------
def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    # 若不希望重复定义计算图上的运算，可通过此方法直接将之前的图加载出来
    saver = tf.train.import_meta_graph("checkpoints/sentiment.ckpt.meta")
    # 此方法从保存的文件中读取已经训练过的神经网络参数
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    # 通过 Tensor 的名称获得对应的张量

    inputs_ = tf.get_default_graph().get_tensor_by_name("inputs:0")
    labels_ = tf.get_default_graph().get_tensor_by_name("labels:0")
    keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
    accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")
    correct_pred = tf.get_default_graph().get_tensor_by_name("correct_pred:0")
    predict_output = tf.get_default_graph().get_tensor_by_name("fully_connected/Sigmoid:0")
    prediction_index = tf.get_default_graph().get_tensor_by_name("prediction_index:0")

    test_output = []
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, 100), 1):
        feed = {inputs_: x,
                labels_: y,
                keep_prob: 1
                }
        output, predict_result, predict_accuracy = sess.run([predict_output, prediction_index, accuracy], feed_dict=feed)
        test_output += output.tolist()


for i in range(0, len(test_output)):
    maxIndex = 0
    for j in range(1, 4):
        if test_output[i][j] > test_output[i][maxIndex]:
            maxIndex = j
    test_output[i] = [0, 0, 0, 0]
    test_output[i][maxIndex] = 1

# Here we got three lists : test_x, test_y, test_output
# Precision and recall can be calculated by them
P_dataIndex = []
I_dataIndex = []
O_dataIndex = []
N_dataIndex = []
for i in range(0, len(test_output)):
    if test_y[i] == [1, 0, 0, 0]:
        P_dataIndex.append(i)
    elif test_y[i] == [0, 1, 0, 0]:
        I_dataIndex.append(i)
    elif test_y[i] == [0, 0, 1, 0]:
        O_dataIndex.append(i)
    elif test_y[i] == [0, 0, 0, 1]:
        N_dataIndex.append(i)
    else:
        print("Error!")

def getMatrixRow(IndexList):
    MaxtrixRow = [0, 0, 0, 0]
    for i in IndexList:
        if test_output[i] == [1, 0, 0, 0]:
            MaxtrixRow[0] += 1
        elif test_output[i] == [0, 1, 0, 0]:
            MaxtrixRow[1] += 1
        elif test_output[i] == [0, 0, 1, 0]:
            MaxtrixRow[2] += 1
        elif test_output[i] == [0, 0, 0, 1]:
            MaxtrixRow[3] += 1
    return  MaxtrixRow


P_row = getMatrixRow(P_dataIndex)
I_row = getMatrixRow(I_dataIndex)
O_row = getMatrixRow(O_dataIndex)
N_row = getMatrixRow(N_dataIndex)

print(P_row)
print(I_row)
print(O_row)
print(N_row)

P_recall = P_row[0] / (P_row[0] + P_row[1] + P_row[2] + P_row[3])
I_recall = I_row[1] / (I_row[0] + I_row[1] + I_row[2] + I_row[3])
O_recall = O_row[2] / (O_row[0] + O_row[1] + O_row[2] + O_row[3])
N_recall = N_row[3] / (N_row[0] + N_row[1] + N_row[2] + N_row[3])

P_precision = P_row[0] / (P_row[0] + I_row[0] + O_row[0] + N_row[0])
I_precision = I_row[1] / (P_row[1] + I_row[1] + O_row[1] + N_row[1])
O_precision = O_row[2] / (P_row[2] + I_row[2] + O_row[2] + N_row[2])
N_precision = N_row[3] / (P_row[3] + I_row[3] + O_row[3] + N_row[3])

print("P ----  recall : " + str(P_recall) + " ||  precision : " + str(P_precision))
print("I ----  recall : " + str(I_recall) + " ||  precision : " + str(I_precision))
print("O ----  recall : " + str(O_recall) + " ||  precision : " + str(O_precision))
print("N ----  recall : " + str(N_recall) + " ||  precision : " + str(N_precision))







