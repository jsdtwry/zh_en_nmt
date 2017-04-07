import numpy as np
import data_util
import tensorflow as tf
from keras.models import Sequential
from keras import layers
import numpy as np
import keras

X_train, Y_train, zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab = data_util.create_dataset('origin_data/spoken.train')

X_test, Y_test = data_util.load_data('origin_data/spoken.test',zh_word2idx, en_word2idx)

def data_padding(x, y, length = 15):
    for i in range(len(x)):
        x[i] = x[i] + (length - len(x[i])) * [zh_word2idx['<pad>']]
        y[i] = [en_word2idx['<go>']] + y[i] + [en_word2idx['<eos>']] + (length-len(y[i])) * [en_word2idx['<pad>']]

data_padding(X_train, Y_train)

data_padding(X_test, Y_test)

input_seq_len = 15
output_seq_len = 17

zh_vocab_size = len(zh_vocab) + 2 # + <pad>, <ukn>
en_vocab_size = len(en_vocab) + 4 # + <pad>, <ukn>, <eos>, <go>

# x, y = [], []
x = np.zeros((len(X_train), input_seq_len, zh_vocab_size), dtype=np.bool)
y = np.zeros((len(X_train), output_seq_len, en_vocab_size), dtype=np.bool)
for index, i in enumerate(X_train):
    x[index] = np.eye(zh_vocab_size)[i]
for index, i in enumerate(Y_train):
    y[index] = np.eye(en_vocab_size)[i]

# x = np.array(x)
# y = np.array(y)

# test_x, test_y = [], []
test_x = np.zeros((len(X_test), input_seq_len, zh_vocab_size), dtype=np.bool)
test_y = np.zeros((len(Y_test), output_seq_len, en_vocab_size), dtype=np.bool)
for index, i in enumerate(X_test):
    test_x[index] = np.eye(zh_vocab_size)[i]
for index, i in enumerate(Y_test):
    test_y[index] = np.eye(en_vocab_size)[i]

#for i in [zh_idx2word[word] for word in test_x[0]]

# test_x = np.array(test_x)
# test_y = np.array(test_y)



np.save('pack.npz',np.array([x, y, test_x, test_y, zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab]))

def gen_batch(x,y,batch_size=64):
    idxs = np.random.choice(len(x), size = batch_size, replace = False)
    b_x = np.array([x[i] for i in idxs])
    b_y = np.array([y[i] for i in idxs])
    return b_x, b_y


RNN = layers.LSTM
HIDDEN_SIZE = 512
BATCH_SIZE = 64
LAYERS = 3

print 'Build model...'

model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(input_seq_len, zh_vocab_size)))
model.add(layers.RepeatVector(output_seq_len))

for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(en_vocab_size)))
model.add(layers.Activation('softmax'))
adam = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=rms)
model.summary()

for iteration in range(1, 20):
#    model.fit(x, y, batch_size=BATCH_SIZE, epochs=10)
    batch_x, batch_y = gen_batch(x, y)
    model.fit(batch_x, batch_y, epochs=10)

model.save('tran.hdf5')


