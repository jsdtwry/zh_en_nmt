import numpy as np
import data_util
import tensorflow as tf
from tensorflow.contrib import rnn

learning_rate = 0.001
training_iter = 1000

batch_size = 64

X_train, Y_train, zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab = data_util.create_dataset('origin_data/spoken.train')

X_test, Y_test = data_util.load_data('origin_data/spoken.test',zh_word2idx, en_word2idx)

def data_padding(x, y, length = 15):
    for i in range(len(x)):
        x[i] = x[i] + (length - len(x[i])) * [zh_word2idx['<pad>']]
        y[i] = [en_word2idx['<go>']] + y[i] + [en_word2idx['<eos>']] + (length-len(y[i])) * [en_word2idx['<pad>']]

data_padding(X_train, Y_train)

data_padding(X_test, Y_test)
'''
print X_train[0]
print Y_train[0]

for ii in [zh_idx2word[i] for i in X_train[0]]:
    print ii,
print

print [en_idx2word[i] for i in Y_train[0]]
'''
input_seq_len = 15
output_seq_len = 17

zh_vocab_size = len(zh_vocab) + 2 # + <pad>, <ukn>
en_vocab_size = len(en_vocab) + 4 # + <pad>, <ukn>, <eos>, <go>

'''
print X_train[0]
x = np.eye(zh_vocab_size)[X_train[0]]
print x.shape
print x
'''
x, y = [], []
for i in X_train:
    x.append(np.eye(zh_vocab_size)[i])
for i in Y_train:
    y.append(np.eye(en_vocab_size)[i])

x = np.array(x)
y = np.array(y)

test_x, test_y = [], []
for i in X_test:
    test_x.append(np.eye(zh_vocab_size)[i])
for i in Y_test:
    test_y.append(np.eye(en_vocab_size)[i])

test_x = np.array(x)
test_y = np.array(y)


def gen_batch(x,y,batch_size=64):
    idxs = np.random.choice(len(x), size = batch_size, replace = False)
    b_x = np.array([x[i] for i in idxs])
    b_y = np.array([y[i] for i in idxs])
    return b_x, b_y


encoder_inputs = tf.placeholder("float", [None, input_seq_len, zh_vocab_size])
decoder_inputs = tf.placeholder("float", [None, output_seq_len, en_vocab_size])

# without softmax, that is too large for GPU training....
n_hidden = 3004

def RNNAD(x, y): 
    x = tf.transpose(x, [1, 0, 2]) 
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, zh_vocab_size])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, input_seq_len, 0)
    
    y = tf.transpose(y, [1, 0, 2]) 
    # Reshaping to (n_steps*batch_size, n_input)
    y = tf.reshape(y, [-1, en_vocab_size])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    y = tf.split(y, output_seq_len, 0)
    
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float64)
    return outputs, y

pred = RNNAD(x, y)

def loss_f(prediction, truth):
    z = 0 
    for step in range(0, input_seq_len):
        z = z + tf.reduce_mean(tf.pow(prediction[step] - truth[step], 2))
    return z

cost = loss_f(pred[0], pred[1])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)

for i in range(training_iters):
    #batch_x, _ = mnist.train.next_batch(batch_size)
    #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    batch_x, batch_y = gen_batch(x, y)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_x})
    if i%100==0:
        train_loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        test_loss = sess.run(cost, feed_dict={x: test_x, y: test_y})
        print i, 'train_loss:', train_loss, 'test_loss:', test_loss
        saver.save(sess, 'model/model.ckpt'+'.'+str(i))

