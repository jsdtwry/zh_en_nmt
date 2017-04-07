import numpy as np
import data_util
import tensorflow as tf
import time
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib

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

input_seq_len = 15
output_seq_len = 17

zh_vocab_size = len(zh_vocab) + 2 # + <pad>, <ukn>
en_vocab_size = len(en_vocab) + 4 # + <pad>, <ukn>, <eos>, <go>

encoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'encoder{}'.format(i)) for i in range(input_seq_len)]
decoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'decoder{}'.format(i)) for i in range(output_seq_len)]

targets = [decoder_inputs[i+1] for i in range(output_seq_len-1)]
# add one more target
targets.append(tf.placeholder(dtype = tf.int32, shape = [None], name = 'last_target'))
target_weights = [tf.placeholder(dtype = tf.float32, shape = [None], name = 'target_w{}'.format(i)) for i in range(output_seq_len)]

# output projection
size = 512
w_t = tf.get_variable('proj_w', [en_vocab_size, size], tf.float32)
b = tf.get_variable('proj_b', [en_vocab_size], tf.float32)
w = tf.transpose(w_t)
output_projection = (w, b)

outputs, states = seq2seq_lib.embedding_attention_seq2seq(
                                            encoder_inputs,
                                            decoder_inputs,
                                            BasicLSTMCell(size),
                                            num_encoder_symbols = zh_vocab_size,
                                            num_decoder_symbols = en_vocab_size,
                                            embedding_size = 100,
                                            feed_previous = False,
                                            output_projection = output_projection,
                                            dtype = tf.float32)

# define our loss function

def sampled_loss(labels, logits):
    return tf.nn.sampled_softmax_loss(
                        weights = w_t,
                        biases = b,
                        labels = tf.reshape(labels, [-1, 1]),
                        inputs = logits,
                        num_sampled = 512,
                        num_classes = en_vocab_size)

loss = seq2seq_lib.sequence_loss(outputs, targets, target_weights, softmax_loss_function = sampled_loss)


def softmax(x):
    n = np.max(x)
    e_x = np.exp(x - n)
    return e_x / e_x.sum()

# feed data into placeholders
def feed_dict(x, y, batch_size = 64):
    feed = {}
    
    idxes = np.random.choice(len(x), size = batch_size, replace = False)
    
    for i in range(input_seq_len):
        feed[encoder_inputs[i].name] = np.array([x[j][i] for j in idxes])
        
    for i in range(output_seq_len):
        feed[decoder_inputs[i].name] = np.array([y[j][i] for j in idxes])
        
    feed[targets[len(targets)-1].name] = np.full(shape = [batch_size], fill_value = en_word2idx['<pad>'])
    
    for i in range(output_seq_len-1):
        batch_weights = np.ones(batch_size, dtype = np.float32)
        target = feed[decoder_inputs[i+1].name]
        for j in range(batch_size):
            if target[j] == en_word2idx['<pad>']:
                batch_weights[j] = 0.0
        feed[target_weights[i].name] = batch_weights
        
    feed[target_weights[output_seq_len-1].name] = np.zeros(batch_size, dtype = np.float32)
    
    return feed

# decode output sequence
def decode_output(output_seq):
    words = []
    for i in range(output_seq_len):
        smax = softmax(output_seq[i])
        idx = np.argmax(smax)
        words.append(en_idx2word[idx])
    return words

# ops and hyperparameters
learning_rate = 5e-3
# learning_rate = 1
batch_size = 64
steps = 1000

# ops for projecting outputs
outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

# training op
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# init op
init = tf.global_variables_initializer()

# forward step
def forward_step(sess, feed):
    output_sequences = sess.run(outputs_proj, feed_dict = feed)
    return output_sequences

# training step
def backward_step(sess, feed):
    sess.run(optimizer, feed_dict = feed)

# let's train the model

# we will use this list to plot losses through steps
losses = []

# save a checkpoint so we can restore the model later 
saver = tf.train.Saver()

print '------------------TRAINING------------------'

path = tf.train.latest_checkpoint('checkpoints')
with tf.Session() as sess:
    sess.run(init)
    
    saver.restore(sess, path)
    
    t = time.time()
    for step in range(steps):
        feed = feed_dict(X_train, Y_train)
            
        backward_step(sess, feed)
        
        if step % 5 == 4 or step == 0:
            loss_value = sess.run(loss, feed_dict = feed)
            print 'step: {}, loss: {}'.format(step, loss_value)
            losses.append(loss_value)
        
        if step % 20 == 19:
            saver.save(sess, 'checkpoints_1/', global_step=step)
            print 'Checkpoint is saved'
            
    print 'Training time for {} steps: {}s'.format(steps, time.time() - t)


# let's test the model

with tf.Graph().as_default():
    
    # placeholders
    encoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'encoder{}'.format(i)) for i in range(input_seq_len)]
    decoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'decoder{}'.format(i)) for i in range(output_seq_len)]

    # output projection
    size = 512
    w_t = tf.get_variable('proj_w', [en_vocab_size, size], tf.float32)
    b = tf.get_variable('proj_b', [en_vocab_size], tf.float32)
    w = tf.transpose(w_t)
    output_projection = (w, b)
    
    # change the model so that output at time t can be fed as input at time t+1
    outputs, states = seq2seq_lib.embedding_attention_seq2seq(
                                                encoder_inputs,
                                                decoder_inputs,
                                                BasicLSTMCell(size),
                                                num_encoder_symbols = zh_vocab_size,
                                                num_decoder_symbols = en_vocab_size,
                                                embedding_size = 100,
                                                feed_previous = True, # <-----this is changed----->
                                                output_projection = output_projection,
                                                dtype = tf.float32)
    
    # ops for projecting outputs
    outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

    # let's translate these sentences     
    # zh_sentences = ["What' s your name", 'My name is', 'What are you doing', 'I am reading a book',\
    #                'How are you', 'I am good', 'Do you speak English', 'What time is it', 'Hi', 'Goodbye', 'Yes', 'No']
    # zh_sentences_encoded = [[zh_word2idx.get(word, 0) for word in zh_sentence.split()] for zh_sentence in zh_sentences]
    
    # padding to fit encoder input
    # for i in range(len(zh_sentences_encoded)):
    #    zh_sentences_encoded[i] += (15 - len(zh_sentences_encoded[i])) * [zh_word2idx['<pad>']]
    
    # restore all variables - use the last checkpoint saved
    zh_sentences_encoded = X_test
    saver = tf.train.Saver()
    path = tf.train.latest_checkpoint('checkpoints_1')
    
    with tf.Session() as sess:
        # restore
        saver.restore(sess, path)
        
        # feed data into placeholders
        feed = {}
        for i in range(input_seq_len):
            feed[encoder_inputs[i].name] = np.array([zh_sentences_encoded[j][i] for j in range(len(zh_sentences_encoded))])
            
        feed[decoder_inputs[0].name] = np.array([en_word2idx['<go>']] * len(zh_sentences_encoded))
        
        # translate
        output_sequences = sess.run(outputs_proj, feed_dict = feed)
        
        # decode seq.
        for i in range(len(zh_sentences_encoded)):
            print '{}.\n--------------------------------'.format(i+1)
            ouput_seq = [output_sequences[j][i] for j in range(output_seq_len)]
            #decode output sequence
            words = decode_output(ouput_seq)
        
            for ii in [zh_idx2word[i] for i in zh_sentences_encoded[i]]:
                print ii,
            print
            
            for i in range(len(words)):
                if words[i] not in ['<eos>', '<pad>', '<go>']:
                    print words[i],
            
            print '\n--------------------------------'








