import os
import codecs
ifilename = "USAAR-DFKI_TRANSAPE-NMT_PRIMARY.txt"
out_dir = "APE_NMT"
mn= "USAAR-DFKI_TRANSAPE-NMT" #METHOD-NAME is the name of your automatic post-editing method.
sn = 1 # SEGMENT-NUMBER is the line number of the plain text target file you are post-editing.
ape_out = "" #the automatic post-edition for the particular segment.

if not os.path.exists(out_dir): os.mkdir(out_dir)
with codecs.open(ifilename, 'r', "utf-8") as fin, codecs.open(out_dir+"/"+ifilename, "w", "utf-8") as fout:

    for line in fin.read().split('\n') :
        ape_out = line.rstrip('\n')

        fout.write(mn+ "\t" + str(sn) +"\t" + ape_out+"\n")
        sn += 1
        fout.flush()