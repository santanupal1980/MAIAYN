import os, sys
import codecs
import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *

itokens1, itokens2, otokens = dd.MakeS2SDictMS('train.tok.ms', dict_file='apevocab.txt')
Xtrain1, Xtrain2, Ytrain = dd.MakeS2SDataMS('train.tok.ms', itokens1, itokens2, otokens, h5_file='ape.train.h5')
Xvalid1, Xvalid2, Yvalid = dd.MakeS2SDataMS('dev.tok.ms', itokens1, itokens2, otokens, h5_file='ape.valid.h5')

print('seq 1 words:', itokens1.num())
print('seq 2 words:', itokens2.num())
print('seq 3 words:', otokens.num())
print('train shapes:', Xtrain1.shape, Xtrain2.shape, Ytrain.shape)
print('valid shapes:', Xvalid1.shape, Xvalid2.shape, Yvalid.shape)

'''
from rnn_s2s import RNNSeq2Seq
s2s = RNNSeq2Seq(itokens,otokens, 256)
s2s.compile('rmsprop')
s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None))
'''

from transformerMS import Transformer, LRSchedulerPerStep, LRSchedulerPerEpoch

d_model = 256
s2s = Transformer(itokens1, itokens2, otokens, len_limit=400, d_model=d_model, d_inner_hid=512, \
                  n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1)

lr_scheduler = LRSchedulerPerStep(d_model, 4000)  # there is a warning that it is slow, however, it's ok.
# lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
model_saver = ModelCheckpoint('ape.model.h5', save_best_only=True, save_weights_only=True)

s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
s2s.model.summary()
try:
    s2s.model.load_weights('ape.model.h5')
    nbest = 1
    if not os.path.exists('results'): os.mkdir('results')
    with codecs.open('dev.tok.ms', 'r', 'utf-8') as fin, codecs.open("results/dev.joint.greedy", "w",
                                                                     "utf-8") as fout, codecs.open("results/dev.joint.beam",
                                                                                                   "w",
                                                                                                   "utf-8") as fout1:
        for line in fin.read().split('\n'):
            values = line.split("\t")
            src = values[0]
            mt = values[1]
            input1_seq = src.split()
            input2_seq = mt.split()
            print('T-O\t' + values[2])
            got_greedy = s2s.decode_sequence_fast(input1_seq, input2_seq)
            print('T-G\t' + got_greedy)
            fout.write(got_greedy + "\n")
            fout.flush()
            pe = s2s.beam_search(input1_seq, input2_seq)
            for got_beam, got_prob in pe[:nbest]:
                print('T-B\t' + got_beam, got_prob)  # y is the predicted probability
                fout1.write(got_beam+"\n")
                fout1.flush()
except:
    print('\n\n Unable to load')
