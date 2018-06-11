import codecs
files = ["processed_org_data/test.bpe.src",
         "processed_org_data/test.bpe.mt",
         "processed_org_data/test.bpe.mt"]

outfile = "processed_org_data/test.ms"

with open(files[0], 'r') as fin1, \
    open(files[1], 'r') as fin2, \
    open(files[2], 'r') as fin3, \
    open(outfile, "w") as fout1:
    for line1, line2, line3  in zip (fin1.read().split('\n'), fin2.read().split('\n'),fin3.read().split('\n'),) :
        fout1.write(line1.rstrip('\n')+ "\t" + line2.rstrip('\n') +"\t" + line3.rstrip('\n')+"\n")
        fout1.flush()