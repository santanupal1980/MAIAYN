import os, sys
import codecs
import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *
from config import config as cfg

itokens1, itokens2, otokens = dd.MakeS2SDictMS(cfg.trn_vocab_corpus, dict_file=cfg.vocab_file) # preparing vocabulary ******* DONT CHANGE FOR FINE TUNE***********
Xtrain1, Xtrain2, Ytrain = dd.MakeS2SDataMS(cfg.training_corpus, itokens1, itokens2, otokens, h5_file=cfg.training_corpus+'.h5', max_len=cfg.len_limit) #prepare data, if fine tuning change both inputs
Xvalid1, Xvalid2, Yvalid = dd.MakeS2SDataMS(cfg.validation_corpus, itokens1, itokens2, otokens, h5_file=cfg.validation_corpus+'.h5', max_len=cfg.len_limit) # prepare validation data

print('seq 1 words:', itokens1.num()) # Number of source vocabulary
print('seq 2 words:', itokens2.num()) # Number of mt vocabulary
print('seq 3 words:', otokens.num()) # Number of target vocabulary

print('train shapes:', Xtrain1.shape, Xtrain2.shape, Ytrain.shape) #Shapes (total corpus size, maximum sentence length)
print('valid shapes:', Xvalid1.shape, Xvalid2.shape, Yvalid.shape)

'''
from rnn_s2s import RNNSeq2Seq
s2s = RNNSeq2Seq(itokens,otokens, 256)
s2s.compile('rmsprop')
s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None))
'''

from transformerMSadd import Transformer, LRSchedulerPerStep, LRSchedulerPerEpoch

d_model = cfg.d_model
s2s = Transformer(itokens1, itokens2, otokens, len_limit=cfg.len_limit, d_model=d_model, d_inner_hid=cfg.d_inner_hid, \
                  n_head=cfg.n_head, d_k=cfg.d_k, d_v=cfg.d_v, layers=cfg.layers, dropout=cfg.dropout)

#lr_scheduler = LRSchedulerPerStep(d_model, cfg.warm_up)  # there is a warning that it is slow, however, it's ok.
#lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
#model_saver = ModelCheckpoint(cfg.output_model+".h5", save_best_only=True, save_weights_only=True)

s2s.compile(Adam(cfg.learning_rate, 0.9, 0.98, epsilon=1e-9))
s2s.model.summary()
try:
    s2s.model.load_weights(cfg.output_model+".h5")
    print("the model has been loaded!")
    nbest = 1
    if not os.path.exists('results'): os.mkdir('results')
    with codecs.open(cfg.test_corpus, 'r', 'utf-8') as fin, \
            codecs.open("results/"+cfg.out_file+".1g", "w", "utf-8") as fout, \
            codecs.open("results/"+cfg.out_file+".1b", "w", "utf-8") as fout1:
        for line in fin.read().split('\n'):
            values = line.split("\t")
            src = values[0]
            mt = values[1]
            input1_seq = src.split()
            input2_seq = mt.split()
            if len(values) == 3:
                print('T-O\t' + values[2])
            got_greedy = s2s.decode_sequence_fast(input1_seq, input2_seq)
            #print('T-G\t' + got_greedy)
            fout.write(got_greedy + "\n")
            fout.flush()

            pe = s2s.beam_search(input1_seq, input2_seq, topk=cfg.beam_size)
            for got_beam, got_prob in pe[:nbest]:
                #print('T-B\t' + got_beam, got_prob)  # y is the predicted probability
                fout1.write(got_beam + "\n")
                fout1.flush()
except Exception as e:
    print('\n\n Unable to load!\n'+ e)
