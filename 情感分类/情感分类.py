from tensorflow import keras
import matplotlib.pyplot as plt

vocab_size = 30000
max_length = 100
embedding_dim = 100
num_classes = 8
units = 64
batch_size = 64
epochs = 5

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

# truncating='pre'——>把前面部分截断 truncating='post'——>把后面部分截断
x_train = keras.preprocessing.sequence.pad_sequences(x_train, max_length, padding='post', truncating='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, max_length, padding='post')
# print(x_train.shape, ' ', y_train.shape)
# print(x_test.shape, ' ', y_test.shape)
# print(x_train[0])

# 构建模型
class RNNModel(keras.Model):

    def __init__(self, units):
        super(RNNModel, self).__init__()

        self.units = units

        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)

        # return_sequences = True——>返回最后一个状态输出，也是我们要输出下一层的

        self.lstm = keras.layers.LSTM(units, return_sequences = True)
        self.lstm_2 = keras.layers.LSTM(64)
        self.dense = keras.layers.Dense(1)

    def call(self, x, training=None, mask=None):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.lstm_2(x)
        x = self.dense(x)
        return x

model = RNNModel(units)

# loss and optimizer
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
#
# model.summary()

# evaluate
# result = model.evaluate(test_data, test_labels)
#
# # output:loss: 0.6751 - accuracy: 0.8002


history_dict = history.history

# draw
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo"——>"blue dot（蓝点）"
plt.plot(epochs, loss, 'bo', label='Training loss')
# "r"——>"solid blue line（红色实线）"
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()