import numpy as np
import tensorflow as tf
from model import Graph, arg
from data_process import encoder_vocab, decoder_vocab

arg.is_training = False

g = Graph(arg)

saver =tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'logs/model')
    while True:
        line = input('输入测试: ')
        if line == 'q':
            break
        try:
            line = line[:-1]
            line = line.lower().replace(',', ' ,').strip('\n').split(' ')
            x = np.array([encoder_vocab.index(pny) for pny in line])
            x = x.reshape(1, -1)
            de_inp = [[decoder_vocab.index('<GO>')]]
            while True:
                y = np.array(de_inp)
                preds = sess.run(g.preds, {g.x: x, g.de_inp: y})
                if preds[0][-1] == decoder_vocab.index('<EOS>'):
                    break
                de_inp[0].append(preds[0][-1])
            ans = ''.join(decoder_vocab[idx] for idx in de_inp[0][1:])
            print(ans)
        except:
            print("error")