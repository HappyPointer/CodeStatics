# coding=utf-8
import tensorflow as tf
import data.cnews_loader
import funcLib
from sys import argv

'''
num1 = argv[1]
num2 = argv[2]
sum = int(num1) + int(num2)
print(sum)
'''
def perdictSentences(sentenceList):
    wordsList, word2id = data.cnews_loader.read_vocab("./data/cnews/cnews.vocab.txt")
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


        sentenceList = funcLib.processEnglish(sentenceList)
        sentenceList = funcLib.transformVec2Id(sentenceList, word2id)
        feed = {inputs_: sentenceList,
                keep_prob: 1}
        predict_index = sess.run(predict_output, feed_dict=feed)

        ResultLabelList = []
        # print("Prediction output : " + str(pred_output))
        for singleIndex in predict_index:
            if funcLib.compare(singleIndex[0],singleIndex[1],singleIndex[2],singleIndex[3]) :
                ResultLabelList.append(("P",singleIndex[0]))
            elif funcLib.compare(singleIndex[1],singleIndex[0],singleIndex[2],singleIndex[3]):
                ResultLabelList.append(("I",singleIndex[1]))
            elif funcLib.compare(singleIndex[2],singleIndex[0],singleIndex[1],singleIndex[3]):
                ResultLabelList.append(("O",singleIndex[2]))
            else:
                ResultLabelList.append(("U",singleIndex[3]))
    return ResultLabelList