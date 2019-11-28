import os
import tensorflow as tf
from model import rnn_model
from poems import process_poems,generate_batch

batch_size = 64
learning_rate = 0.01
check_points_dir = './model/'
file_path = './data/poems.txt'
model_prefix = 'poems'
epochs = 50

start_token = 'G'
end_token = 'E'


def run_training():
    if not os.path.exists(os.path.dirname(check_points_dir)):
        os.mkdir(os.path.dirname(check_points_dir))
    if not os.path.exists(check_points_dir):
        os.mkdir(check_points_dir)

    poems_vector, word_to_int, vocabularies = process_poems(file_path)
    batches_inputs, batches_outputs = generate_batch(batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(check_points_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("Restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('Start training...')
        try:
            for epoch in range(start_epoch, epochs):
                n = 0
                n_chunk = len(poems_vector) // batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))

                if epoch % 6 == 0:
                    # saver.save(sess, './model/', global_step=epoch)
                    saver.save(sess, os.path.join(check_points_dir, model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(check_points_dir, model_prefix), global_step=epoch)
            print('Last epoch were saved, next time will start from epoch {}.'.format(epoch))

# 开启训练
run_training()