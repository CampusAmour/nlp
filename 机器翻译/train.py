import tensorflow as tf
import os
from model import Graph
from data_process import encoder_inputs, decoder_inputs, decoder_targets, encoder_vocab, decoder_vocab, get_batch


def create_hparams():
    params = tf.contrib.training.HParams(
        num_heads=8,
        num_blocks=6,
        # vocab
        input_vocab_size=50,
        label_vocab_size=50,
        # embedding size
        max_length=100,
        hidden_units=512,
        dropout_rate=0.2,
        lr=0.0003,
        is_training=True)
    return params


arg = create_hparams()
arg.input_vocab_size = len(encoder_vocab)
arg.label_vocab_size = len(decoder_vocab)


epochs = 200
batch_size = 64

g = Graph(arg)

saver =tf.train.Saver()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    if os.path.exists('logs/model.meta'):
        saver.restore(sess, 'logs/model')
    writer = tf.summary.FileWriter('tensorboard/transformer', tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch_num = len(encoder_inputs) // batch_size
        batch = get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size)
        for i in range(batch_num):
            encoder_input, decoder_input, decoder_target = next(batch)
            feed = {g.x: encoder_input, g.y: decoder_target, g.de_inp:decoder_input}
            cost,_ = sess.run([g.mean_loss,g.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
        print('epochs', k+1, ': average loss = ', total_loss/batch_num)
    saver.save(sess, 'logs/model')
    writer.close()