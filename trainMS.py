import os, sys
import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *
from config import config as cfg
import tensorflow as tf
import keras
import  matplotlib.pyplot as plt

serverConfig = tf.ConfigProto(device_count = { 'GPU' : 1, 'CPU' : 56})
sess = tf.Session(config=serverConfig)
keras.backend.set_session(sess)

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

lr_scheduler = LRSchedulerPerStep(d_model, cfg.warm_up)  # there is a warning that it is slow, however, it's ok.
# lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
model_saver = ModelCheckpoint(cfg.output_model+".h5", save_best_only=True, save_weights_only=True)

s2s.compile(Adam(cfg.learning_rate, 0.9, 0.98, epsilon=1e-9))
s2s.model.summary()
try:
    s2s.model.load_weights(cfg.output_model+".h5")
except:
    print('\n\nnew model')
history = s2s.model.fit([Xtrain1, Xtrain2, Ytrain], None, batch_size=cfg.batch_size, epochs=cfg.epochs, \
              validation_data=([Xvalid1, Xvalid2, Yvalid], None), \
              callbacks=[lr_scheduler, model_saver])
history_dict = history.history
trn_loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['accu']
val_acc = history_dict['val_accu']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, trn_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
