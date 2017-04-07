import numpy as np
from collections import Counter


'''generate translation pairs from original data'''
def gen_sentences(in_file, out_file_zh, out_file_en):
    f_list = file(in_file).readlines()
    out_zh = file(out_file_zh, 'w')
    out_en = file(out_file_en, 'w')
    for i, line in enumerate(f_list):
        if i%2==0:
            out_zh.write(line)
        else:
            out_en.write(line)
            
'''zh-en translation generate word dictionary and training data'''
def create_dataset(in_file):
    f_list = file(in_file).readlines()
    zh_line, en_line = [], []
    for i, line in enumerate(f_list):
        if i%2==0:
            zh_line.append(line[:-1])
        else:
            en_line.append(line[:-1])
    en_vocab_dict = Counter(word.strip(',." ;:)(][!-') for sentence in en_line for word in sentence.split())
    zh_vocab_dict = Counter(word.strip(',." ;:)(][!-') for sentence in zh_line for word in sentence.split())
        
    en_vocab = map(lambda x: x[0], sorted(en_vocab_dict.items(), key = lambda x: -x[1]))
    zh_vocab = map(lambda x: x[0], sorted(zh_vocab_dict.items(), key = lambda x: -x[1]))
    
    en_vocab = en_vocab[:3000]
    zh_vocab = zh_vocab[:3000]
    
    start_idx = 2
    zh_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(zh_vocab)])
    zh_word2idx['<ukn>'] = 0
    zh_word2idx['<pad>'] = 1

    zh_idx2word = dict([(idx, word) for word, idx in zh_word2idx.iteritems()])
    
    start_idx = 4
    en_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(en_vocab)])
    en_word2idx['<ukn>'] = 0
    en_word2idx['<go>']  = 1
    en_word2idx['<eos>'] = 2
    en_word2idx['<pad>'] = 3
    
    en_idx2word = dict([(idx, word) for word, idx in en_word2idx.iteritems()])
    
    x = [[zh_word2idx.get(word.strip(',." ;:)(][!'), 0) for word in sentence.split()] for sentence in zh_line]
    y = [[en_word2idx.get(word.strip(',." ;:)(][!'), 0) for word in sentence.split()] for sentence in en_line]

    X = []
    Y = []
    for i in range(len(x)):
        n1 = len(x[i])
        n2 = len(y[i])
        n = n1 if n1 < n2 else n2  
        if abs(n1 - n2) <= 0.3 * n:
            if n1 <= 15 and n2 <= 15: 
                X.append(x[i])
                Y.append(y[i])
    '''test
    print len(X),len(Y), len(X[0]), X[0], len(Y[0]), Y[0]
    for ii in [zh_idx2word[i] for i in X[0]]:
        print ii,
    print
    print [en_idx2word[i] for i in Y[0]]
    '''
    return X, Y, zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab
    #print en_idx2word

'''load data with word dictionary'''
def load_data(in_file, zh_word2idx, en_word2idx):
    f_list = file(in_file).readlines()
    zh_line, en_line = [], []
    for i, line in enumerate(f_list):
        if i%2==0:
            #zh_line.append([zh_word2idx[word.strip(',." ;:)(][?!-')] for word in line[:-1].split()])
            each = []
            for word in line[:-1].split():
                if zh_word2idx.has_key(word):
                    each.append(zh_word2idx[word.strip(',." ;:)(][!-')])
                else:
                    each.append(0)
            zh_line.append(each)
        else:
            #en_line.append([en_word2idx[word.strip(',." ;:)(][?!-')] for word in line[:-1].split()])
            each = []
            for word in line[:-1].split():
                if en_word2idx.has_key(word):
                    each.append(en_word2idx[word.strip(',." ;:)(][!-')])
                else:
                    each.append(0)
            en_line.append(each)
    
    return zh_line, en_line

def data_padding(x, y, zh_word2idx, en_word2idx, length = 15):
    for i in range(len(x)):
        x[i] = x[i] + (length - len(x[i])) * [zh_word2idx['<pad>']]
        y[i] = [en_word2idx['<go>']] + y[i] + [en_word2idx['<eos>']] + (length-len(y[i])) * [en_word2idx['<pad>']]

'''save data and word dict to numpy default store'''
def data2np():
    X_train, Y_train, zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab = create_dataset('origin_data/spoken.train')
    X_test, Y_test = load_data('origin_data/spoken.test',zh_word2idx, en_word2idx)
    data_padding(X_train, Y_train, zh_word2idx, en_word2idx)
    data_padding(X_test, Y_test, zh_word2idx, en_word2idx)
    input_seq_len = 15
    output_seq_len = 17
    zh_vocab_size = len(zh_vocab) + 2 # + <pad>, <ukn>
    en_vocab_size = len(en_vocab) + 4 # + <pad>, <ukn>, <eos>, <go>
    x = np.zeros((len(X_train), input_seq_len, zh_vocab_size), dtype=np.bool)
    y = np.zeros((len(X_train), output_seq_len, en_vocab_size), dtype=np.bool)
    for index, i in enumerate(X_train):
        x[index] = np.eye(zh_vocab_size)[i]
    for index, i in enumerate(Y_train):
        y[index] = np.eye(en_vocab_size)[i]

    test_x = np.zeros((len(X_test), input_seq_len, zh_vocab_size), dtype=np.bool)
    test_y = np.zeros((len(Y_test), output_seq_len, en_vocab_size), dtype=np.bool)
    for index, i in enumerate(X_test):
        test_x[index] = np.eye(zh_vocab_size)[i]
    for index, i in enumerate(Y_test):
        test_y[index] = np.eye(en_vocab_size)[i]
    
    np.save('spoken_data',np.array([x, y, test_x, test_y, zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab]))



def test():
    X_train, Y_train, zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab = create_dataset('origin_data/spoken.train')
    X_test, Y_test = load_data('origin_data/spoken.test',zh_word2idx, en_word2idx)
    
    for ii in [zh_idx2word[i] for i in X_test[11]]:
        print ii,
    print
    print [en_idx2word[i] for i in Y_test[11]]

data2np()

#test()
#gen_sentences('origin_data/spoken.train','data/train.zh','data/train.en')
#gen_sentences('origin_data/spoken.valid','data/valid.zh','data/valid.en')
#gen_sentences('origin_data/spoken.test','data/test.zh','data/test.en')
