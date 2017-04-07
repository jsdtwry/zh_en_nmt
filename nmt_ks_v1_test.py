import numpy as np
from keras.models import load_model

# a = np.load('pack.npz.npy')

[x, y, test_x, test_y, zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab] = np.load('pack.npz.npy')

model = load_model('tran.hdf5')

# print test_x.argmax(2).shape
# print test_x.argmax(2)[0]
# print test_x.shape

t_x = test_x.argmax(2)
t_y = test_y.argmax(2)
print t_y.shape
print test_y.shape
preds = model.predict_classes(test_x, verbose=0)

print test_x.shape
print preds.shape

for i in range(100):
    guess = preds[i]
    
    print i,
    for ii in [zh_idx2word[word] for word in t_x[i]]:
        print ii,
    print
    
    print guess
    print i,
    for ii in [en_idx2word[word] for word in guess]:
        print ii,
    print



