class config:
    trn_vocab_corpus = "train.tok.ms"
    training_corpus = "train.tok.ms"
    validation_corpus = "dev.tok.ms"
    vocab_file = "vocab.txt" # do not change in case of finetuning
    #train_h5 = training_corpus+".h5"
    #valid_h5 = validation_corpus+".h5"
    output_model = "ape.model"

    ###############Model hyper-parameter########################
    d_model = 256
    len_limit = 400
    d_inner_hid = 512
    n_head = 4
    d_k = 64
    d_v = 64
    layers = 2
    dropout = 0.1
    warm_up = 4000
    batch_size = 64
    epochs = 30
    learning_rate = 0.001
