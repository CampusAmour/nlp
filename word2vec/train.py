import time
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from createData import vocabulary_size, data, reverse_dictionary
from sklearn.manifold import TSNE

batch_size = 128
embedding_size = 256 #单词转化为稠密词向量的维度
num_sampled = 64  # 训练时用来做负样本的噪声单词的数量
epochs = 10  # 迭代轮数
window_size = 10  # 窗口大小


# 生成Word2Vec的训练样本，使用skip-gram模式
# 获得input word的上下文单词列表
def generate_targets(data, index, window_size=5):
    '''
    parameter
    words: 单词列表
    index: input word的索引号
    window_size: 窗口大小,默认为5
    '''
    # 需考虑input word前面单词不够的情况
    start_point = index - window_size if (index - window_size) > 0 else 0
    end_point = index + window_size
    # output words(即窗口中的上下文单词)
    targets = set(data[start_point: index] + data[index + 1: end_point + 1])
    return list(targets)


def generate_batches(data, batch_size, window_size=5):
    '''
    构造一个获取batch的生成器
    '''
    n_batches = len(data) // batch_size

    # 仅取full batches
    batch_words = data[: n_batches * batch_size]
    words_length = len(batch_words)
    for index in range(0, words_length, batch_size):
        x, y = [], []
        batch = data[index: index + batch_size]
        for i in range(batch_size):
            batch_x = batch[i]
            batch_y = generate_targets(batch, i, window_size)
            # 由于一个input word会对应多个output word，因此需要长度统一
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y

# generate_batches(words, 64, window_size=5)

# 开始定义Skip-Gram Word2Vec模型的网络结构
with tf.name_scope('input'):
    inputs = tf.placeholder(tf.int32, [None], name='inputs')
    labels = tf.placeholder(tf.int32, [None, None], name='labels')

with tf.name_scope('embedding'):
    #单词大小为50000，向量维度为128，随机采样在（-1，1）之间的浮点数
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],-1.0,1.0), name='embeddings')
    # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素
    #使用tf.nn.embedding_lookup()函数查找train_inputs对应的向量embed
    embed = tf.nn.embedding_lookup(embeddings, inputs, name='embed')

with tf.name_scope('loss'):
    W = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=0.1), name='weights')
    b = tf.Variable(tf.zeros(vocabulary_size), name='biases')
    # 计算negative sampling下的损失
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(W, b, labels, embed, num_sampled, vocabulary_size), name='loss')
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(3e-5).minimize(loss)

# 验证训练出的相近语义的词
#################################################
# 抽取10个词
# 从不同位置各选5个单词
valid_size = 10
valid_window = 100

# random.sample, 从list中随机获取valid_size // 2个元素，作为一个片断返回
valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
valid_examples = np.append(valid_examples,
                           random.sample(range(1000, 1000 + valid_window), valid_size // 2))

valid_size = len(valid_examples)
# 验证单词集
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# 计算每个词向量的模并进行单位化
normalized_embeddings = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embedding = embeddings / normalized_embeddings
# 查找验证单词的词向量
valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
# 计算余弦相似度
similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
#################################################

# 可视化
def visualization(embed_matrix, save_filename):
    tsne = TSNE()
    visual_words = 200  # 可视化词语数量
    embed_tsne = tsne.fit_transform(embed_matrix[: visual_words, :])
    plt.figure(figsize=(20, 20))
    for index in range(visual_words):
        plt.scatter(*embed_tsne[index, :], color='steelblue')
        plt.annotate(reverse_dictionary[index], (embed_tsne[index, 0], embed_tsne[index, 1]), alpha=0.7)
    plt.savefig(save_filename)

# 初始化变量
init = tf.global_variables_initializer()

# 保存模型
saver = tf.train.Saver()

# 申请一个空的all_loss列表, 存放所有的loss
all_loss = []

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./logs', sess.graph)
    iteration = 1

    # 由于在训练过程中每次是更新一行embedding矩阵, 因此误差应是训练一定iteration后的平均loss
    # mini_batch_loss指的就是一定iteration(50)后的总loss
    mini_batch_loss = 0

    start_time = time.time()
    for epoch in range(epochs):
        batches = generate_batches(data, batch_size, window_size)

        for x, y in batches:
            # label:[[17983]
            #       [50240]
            #       [24038]
            #       [56151]
            #       [60728]]
            feed = {inputs: x, labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([loss, train], feed_dict=feed)
            # 将每次迭代的损失加起来
            mini_batch_loss += train_loss
            if iteration % 500 == 0:
                print("Epoch {}/{}, ".format(epoch + 1, epochs),
                      "Iteration: {}, ".format(iteration),
                      "Average training loss: {:.4f}, ".format(mini_batch_loss / 500),
                      "Cost time: {:.1f} s.".format(time.time() - start_time))
                all_loss.append([mini_batch_loss / 500, iteration])
                mini_batch_loss = 0
            if iteration % 10000 == 0:
                # 计算similarity
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = data[valid_examples[i]]
                    top_k = 8  # 取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = '最相似单词有: [%s]' % reverse_dictionary[valid_word]
                    for k in range(top_k):
                        close_word = data[nearest[k]]
                        log = '%s %s,' % (log, reverse_dictionary[close_word])
                    print(log)
            if iteration % 100000 == 0:
                # 拿到单位化embedding矩阵参数
                embed_matrix = sess.run(normalized_embedding)
                save_filename = "./pictures/tsne" + str(iteration) + ".png"
                visualization(embed_matrix, save_filename)
            iteration += 1

    # 保存模型
    print("Saving model...")
    saver.save(sess, './net/word2vec.ckpt')
    # 拿到单位化embedding矩阵参数
    embed_matrix = sess.run(normalized_embedding)

# 最后再来一次可视化
save_filename = "./pictures/tsne_final.png"
visualization(embed_matrix, save_filename)

# 将all_loss写入文件,方便后面进行loss可视化操作
with open('./all_loss', 'w') as f:
    for loss in all_loss:
        # rount保留小数点后3位
        f.writelines(str(round(loss[0], 3)) + ' ' + str(loss[1]) + '\n')