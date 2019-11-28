import jieba
import numpy as np

with open('data.txt', 'r', encoding='utf8') as f:
    data = f.readlines()

inputs = []
outputs = []

for line in data:
    [en, ch] = line.strip('\n').split('\t')
    inputs.append(en.replace(',',' ,')[:-1].lower())
    outputs.append(ch[:-1])

inputs = [en.split(' ') for en in inputs]

del data

# print(inputs[:10])

outputs = [[char for char in jieba.cut(line) if char != ' '] for line in outputs]


def get_vocab(data, init=['<PAD>']):
    vocab = init
    for line in data:
        for char in line:
            if char not in vocab:
                vocab.append(char)
    return vocab

SOURCE_CODES = ['<PAD>']
TARGET_CODES = ['<PAD>', '<GO>', '<EOS>']
encoder_vocab = get_vocab(inputs, init=SOURCE_CODES)
decoder_vocab = get_vocab(outputs, init=TARGET_CODES)


encoder_inputs = [[encoder_vocab.index(word) for word in line] for line in inputs]
decoder_inputs = [[decoder_vocab.index('<GO>')] + [decoder_vocab.index(word) for word in line] for line in outputs]
decoder_targets = [[decoder_vocab.index(word) for word in line] + [decoder_vocab.index('<EOS>')] for line in outputs]

del inputs, outputs


def get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size=4):
    batch_num = len(encoder_inputs) // batch_size
    for k in range(batch_num):
        begin = k * batch_size
        end = begin + batch_size
        en_input_batch = encoder_inputs[begin:end]
        de_input_batch = decoder_inputs[begin:end]
        de_target_batch = decoder_targets[begin:end]
        max_en_len = max([len(line) for line in en_input_batch])
        max_de_len = max([len(line) for line in de_input_batch])
        en_input_batch = np.array([line + [0] * (max_en_len-len(line)) for line in en_input_batch])
        de_input_batch = np.array([line + [0] * (max_de_len-len(line)) for line in de_input_batch])
        de_target_batch = np.array([line + [0] * (max_de_len-len(line)) for line in de_target_batch])
        yield en_input_batch, de_input_batch, de_target_batch

batch = get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size=4)