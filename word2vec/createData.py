import collections

filename = "./data/text8"

# 构建词汇表，并统计每个单词出现的频数，同时用字典的形式进行存储取，频数排名前50000的单词
vocabulary_size = 50000


# 读取文件中单词数目
def read_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(line)
        words = data[0].split(" ", -1)
    return words[1:] # 之前的words[0]是一个空字符串''


def build_dataset(words):
    count = []

    # 载入停用词
    stop_words = []
    # stop_word2中停用词为138
    with open('./data/stop_word2', 'r') as f:
        for line in f:
            stop_word, _ = line.split('\n')
            stop_words.append(stop_word)
    # 若words中有停用词，则将该词替换成'unknown'
    # words中含有620224个停用词
    words_count = len(words)
    for i in range(words_count):
        if words[i] in stop_words:
            words[i] = 'unknown'

    # most_common作用: 保留前top(50000)单词及数量
    count.extend(collections.Counter(words).most_common(vocabulary_size))
    # 这里将'unknown'换到第一位置上去
    # 形如: [('unknown', 621390), ('the', 1061396), ('of', 593677), ('and', 416629)]
    count[0], count[1] = count[1], count[0]
    # 形如: [['unknown', 621390], ('the', 1061396), ('of', 593677), ('and', 416629)]
    count[0] = list(count[0])

    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # print(dictionary['unknown']) 'unknown'所对应的是0, the-1, of-2

    data = []
    unk_count = 0  # 准备统计top50000以外的单词的个数

    # 对原文本进行vocabulary到int的转换
    # 将words中对应的单词换成词汇表中所对应的索引
    # 若该词在词汇表中, 则让其编号等于该词在list表中的编号, 否则让其等于‘unknown’编号0
    # data形如[5184, 3032, 11, 6, 180, 2, 3085, 44]
    for word in words:
        # 对于其中每一个单词，首先判断是否出现在字典当中
        if word in dictionary:
            # 如果已经出现在字典中，则转为其编号
            index = dictionary[word]
        else:
            # 如果不在字典，则转为‘unknown’编号0
            index = 0
            unk_count += 1
        data.append(index)  # 此时单词已经转变成对应的编号
    # print(data)

    # 将统计好的unknown的单词数，填入count中
    count[0][1] += unk_count

    # 将字典进行翻转 {'the': 1, 'of': 2, 'and': 3} ---> {1: 'the', 2: 'of', 3: 'and'}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

words = read_file(filename)
# print(words)
data, count, dictionary, reverse_dictionary = build_dataset(words)
print("总的单词个数：",len(words))
del words # 不让其再占用内存
print("词汇表单词个数：",len(dictionary))