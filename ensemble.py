import codecs
from collections import defaultdict
# takes the output files of 6 models(3 normal nmt and 3 convolutional nmt)

total_sentences = 1020 # input1

model_outputs = ["./translations_on_test/translate_avg_MS_NMT/average-0",
                 "./translations_on_test/translate_avg_MS_NMT_fine_tune_test/average-0",
                 "./translations_on_test/translate_avg_SS_NMT/average-0",
                 "./translations_on_test/translate_avg_SS_NMT_fine_tune_test/average-0",
                 "./translations_on_test/translate_MS_NMT/model_epoch_53_gs_22207",
                 "./translations_on_test/translate_MS_NMT_fine_tune/model_epoch_15_gs_690",
                 "./translations_on_test/translate_SS_NMT/model_epoch_60_gs_25140",
                 "./translations_on_test/translate_SS_NMT_fine_tune/model_epoch_10_gs_460"
                 ] # input2

# dic = defaultdict(list)
# for i, fname in enumerate(model_outputs):
#     #print([i , fname])
#     list1=[]
#     with codecs.open(fname, 'r', 'utf-8') as fin:
#         for line in fin.read().split('\n'):
#             list1.append(line)
#     dic[i].append(list1)


a = open(model_outputs[0])
b = open(model_outputs[1])
c = open(model_outputs[2])
d = open(model_outputs[3])
e = open(model_outputs[4])
f = open(model_outputs[5])

output = open('output_ens.txt', 'w')
# accuracies based on BLEU noted by the 6 models, used in weighted ensembling

weights = [0.74864, 0.75012, 0.72612, 0.73296, 0.74846, 0.75006, 0.72757, 0.73317] # input3: based on BLEU scores
for i in range(total_sentences):
    translation1 = a.readline()

    translation2 = b.readline()
    translation3 = c.readline()
    translation4 = d.readline()
    translation5 = e.readline()
    translation6 = f.readline()
    translation1 = translation1.rstrip('\n')
    print(i, translation1)
    translation2 = translation2.rstrip('\n')
    translation3 = translation3.rstrip('\n')
    translation4 = translation4.rstrip('\n')
    translation5 = translation5.rstrip('\n')
    translation6 = translation6.rstrip('\n')

    list_model = [translation1, translation2, translation3, translation4, translation5, translation6]
    dict1 = {}
    i = 0
    for l1 in list_model:
        if l1 in dict1:
            dict1[l1] = dict1[l1] + (1 * weights[i])  # weighted ensembling
        # dict1[l1]=dict1[l1]+1		#normal ensembling
        else:
            dict1[l1] = (1 * weights[i])  # weighted ensembling
        # dict1[l1]=1		#normal ensembling
        i += 1
    max_index = 0
    for key in dict1:
        if (dict1[key] > max_index):  # maximum frequency is considered
            max_index = dict1[key]
            str_index = key
    # print(dict1)
    output.write(str_index)  # creating a new result file from the respective maximum frequencies
    output.write('\n')
    output.flush()
