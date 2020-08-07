# encoding=utf-8

# 评估： ---------------------------------------------------------------------------
# 当前该代码的模型建立策略：
# 预处理方面仅进行了简单的特殊字符过滤与词干提取，未添加停用词方法
# 模型方面目前采用2层Bi-LSTM网络，在6000个数据训练50代后，模型准确率达到饱和
# 输入数据中分为 P,I,O,U 四个标签，数据量比例约为 3:1:1:1
# 目前该模型理论准确度大约为82%，在实际测试中效果基本符合要求
# 训练中在训练集上的准确度达到了97%，过拟合现象似乎无法缓和，但是模型仍有一定的效果
# ---------------------------------------------------------------------------------


import tensorflow as tf
import numpy as np
import data.cnews_loader
import funcLib
import random
import matplotlib.pyplot as plt

print("Version: ", tf.__version__)

# 读取文本数据，并对文本进行预处理
train_x, train_y = data.cnews_loader.read_file("./data/cnews/cnews.train.txt")
test_x, test_y = data.cnews_loader.read_file("./data/cnews/cnews.test.txt")
val_x, val_y = data.cnews_loader.read_file("./data/cnews/cnews.val.txt")

# 建立词向量
wordsList, word2id = data.cnews_loader.read_vocab("./data/cnews/cnews.vocab.txt")
train_x = funcLib.transformVec2Id(train_x, word2id)
test_x = funcLib.transformVec2Id(test_x, word2id)
val_x = funcLib.transformVec2Id(val_x, word2id)

train_y = funcLib.transformLabel2Onehot_with4classes(train_y)
test_y = funcLib.transformLabel2Onehot_with4classes(test_y)
val_y = funcLib.transformLabel2Onehot_with4classes(val_y)


# 打乱数据集的顺序
def shuffleDataSet(data_X, data_Y):
    wholeList = []
    for i in range(0, len(data_X)):
        wholeList.append([data_X[i], data_Y[i]])
    random.shuffle(wholeList)

    shuffled_x = []
    shuffled_y = []
    for i in range(0, len(wholeList)):
        shuffled_x.append(wholeList[i][0])
        shuffled_y.append(wholeList[i][1])
    return [shuffled_x, shuffled_y]


# 打乱数据集中的数据顺序
train_x, train_y = shuffleDataSet(train_x, train_y)
test_x, test_y = shuffleDataSet(test_x, test_y)
val_x, val_y = shuffleDataSet(val_x, val_y)


# Build the Neural Network --------------------------------------------------------------------------------------
lstm_size = 128
lstm_layers = 2
num_classes = 4
batch_size = 250
learning_rate = 0.001
drop_out = 0.5
epochs = 50


# Create TF Placeholders for the Neural Network.
n_words = len(word2id)
# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, num_classes], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Embedding
embed_size = 64   # Size of the embedding vectors (number of units in the embedding layer)
with graph.as_default():
    # tf.random_uniform((n_words, embed_size), -1, 1) 产生一个 -1 - 1 之间，形状为 n_words*embed_size 的 tensor 对象
    # embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))  # n_words 是总单词数
    embedding = tf.Variable(tf.random_normal([n_words, embed_size]), dtype=tf.float32)
    embed = tf.nn.embedding_lookup(embedding, inputs_) # para: sourceData, lookup



# Build RNN Cell and Initialize
# Stack one or more LSTMCells in a MultiRNNCell.
with graph.as_default():
    def lstm_cell():
        cell = tf.contrib.rnn.LSTMCell(lstm_size,   # lstm_size = 256 为可调参数，一层神经网络中神经元个数，即隐藏神经元数量
                                       # 均匀分布的张量的初始化器，初始化模型参数
                                       # initializer=tf.random_normal_initializer(0, 0.001, dtype=tf.float32), # 高斯初始化器
                                       initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2),
                                       # 返回存储LSTM单元的state_size,zero_state和output state的元组。按顺序存储两个元素(c,h) c是隐藏状态，h是输出
                                       state_is_tuple=True)

        drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return drop

    cell_fw = [lstm_cell() for _ in range(lstm_layers)]
    cell_bw = [lstm_cell() for _ in range(lstm_layers)]

    output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=embed, dtype=tf.float32)


# Output Layer 全连接层的构建及优化器设置、精度输出
with graph.as_default():
    # 建立 Output Layer 全连接

    predict_output = tf.contrib.layers.fully_connected(output[:, -1], num_classes, activation_fn=tf.sigmoid)  # 获得分类结果，多分类使用其它激活函数
    prediction_index = tf.cast(tf.argmax(predict_output, axis=1), tf.int32, name='prediction_index')

    correct_pred = tf.equal(prediction_index, tf.cast(tf.argmax(labels_, 1), tf.int32), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    # cost = tf.losses.mean_squared_error(labels_, predictions)

    # 定义优化器
    # label_reshape = tf.cast(tf.reshape(labels_, [-1]), tf.int32)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=tf.cast(predict_output, tf.float32))
    )
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


with graph.as_default():
    saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95 # 占用GPU90%的显存
with tf.Session(config=config, graph=graph) as sess:
    # Record the accuracies in order to plot later
    test_acc_list = []
    val_acc_list = []
    max_iteration = 0

    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):  # epochs = 5 为可调参数
        # 这里 x 包括 batch_size 个词向量，y 是 batch_size 个 0/1 数值
        # x, y 均为 ndarray 变量
        for x, y in get_batches(train_x, train_y, batch_size):
            feed = {inputs_: x,
                    labels_: y,
                    keep_prob: drop_out,
                    }
            returned_accuracy, loss, pre_index, _ = sess.run([accuracy, cross_entropy, prediction_index, optimizer], feed_dict=feed)

            if iteration % 5 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration))
                print("Accuracy on training set : " + str(returned_accuracy))

            if iteration % 20 == 0:
                val_acc = []
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y,
                            keep_prob: 1
                            }
                    batch_acc, val_state = sess.run([accuracy, output], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Accuracy on validation set is : {:.3f} ***********".format(np.mean(val_acc)))

                # data recorded will be used to plot later
                val_acc_list.append(np.mean(val_acc))
                test_acc_list.append(np.mean(returned_accuracy))
                max_iteration += 20

                # test set ------------------------------------------------
                val_acc = []
                for x, y in get_batches(test_x, test_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y,
                            keep_prob: 1
                            }
                    batch_acc, val_state = sess.run([accuracy, output], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Accuracy on testing set is : {:.3f} ***********".format(np.mean(val_acc)))
                # ------------------------------------------------

            iteration += 1
    saver.save(sess, "checkpoints/sentiment.ckpt")


    # ploting
    plot_x = list(range(10, max_iteration + 10, 20))
    plt.plot(plot_x, test_acc_list, label='train set')
    plt.plot(plot_x, val_acc_list, label='validation set')
    plt.legend()
    plt.show()

test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y,
                keep_prob: 1
                }
        batch_acc, test_state = sess.run([accuracy, output], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
