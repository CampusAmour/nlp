import time
import numpy as np
import tensorflow as tf
from model import rnn_model
from poems import process_poems

start_token = 'G'
end_token = 'E'

batch_size = 64
learning_rate = 0.0002
check_points_dir = './model/'
file_path = './data/poems.txt'
model_prefix = 'poems'
epochs = 50

def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]

def gen_poem(begin_word):
    batch_size = 1
    print('Loading corpus from %s' % file_path)
    poems_vector, word_int_map, vocabularies = process_poems(file_path)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        checkpoint = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, checkpoint)
        # saver.restore(sess, './model/-24')

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        poem = ''
        while word != end_token:
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
        # word = words[np.argmax(probs_)]
        return poem

def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    print(poem_sentences)
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')

begin_word = input('请输入起始字:')
# begin_word = '重'
start_time = time.time()
poem2 = gen_poem(begin_word)
pretty_print_poem(poem2)
print("time: %4.4f" % (time.time() - start_time))