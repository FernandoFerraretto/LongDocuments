import os
# from numpy import loadtxt
import numpy as np
from numpy import savetxt
# from datasets import load_dataset

os.system('clear')


# f_main = 'dataset_msmarco/triples.train.small.tsv'

# #carrega dos arquivos csv
# set_main = load_dataset('csv', delimiter = '\t', data_files=f_main)
# print(set_main.keys())

#for item in set_main['train']:

import csv
f_main = 'dataset_msmarco/triples.train.small.tsv'
tsv_file = open(f_main,encoding='utf-8')
read_tsv = csv.reader(tsv_file, delimiter="\t")

#csv_headings = next(read_tsv)
#print(csv_headings[0])
#print(len(csv_headings[0].split()))

i=0
max_itens = 22400
with open('dataset_msmarco/V02/msmarco_NoNoise.csv', 'wt') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n', )
    writer.writerow(['label','title','text'])
    for row in read_tsv:
        #   amostra par: 1 | TITULO + RUIDO (NEG) | POS
        # amostra impar: 0 | TITULO + RUIDO (POS) | NEG
        if i%2 == 0:
            ttitle = row[0]
            ttext = row[1]
            tnoise = row[2]
            tlabel = 1
        else:
            ttitle = row[0]
            ttext = row[2]
            tnoise = row[1]
            tlabel = 0

        writer.writerow([tlabel, ttitle,ttext])
        print(len(ttitle.split())+len(ttext.split()) ) 

        if i> max_itens -2:
            break
        else:
            i = i +1

        
tsv_file.close()
