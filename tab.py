import codecs
files = ["data/dev.bpe.src",
         "data/dev.bpe.mt",
         "data/dev.bpe.pe"]

outfile = "data/dev.ms"

with open(files[0], 'r', encoding='utf8') as fin1, \
    open(files[1], 'r', encoding='utf8') as fin2, \
    open(files[2], 'r', encoding='utf8') as fin3, \
    open(outfile, "w", encoding='utf8') as fout1:
    for line1, line2, line3  in zip (fin1.read().split('\n'), fin2.read().split('\n'),fin3.read().split('\n'),) :
        fout1.write(line1.rstrip('\n')+ "\t" + line2.rstrip('\n') +"\t" + line3.rstrip('\n')+"\n")
        fout1.flush()