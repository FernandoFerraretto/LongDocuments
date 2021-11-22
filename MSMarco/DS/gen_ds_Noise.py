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
read_tsv2 = csv.reader(tsv_file, delimiter="\t")


#csv_headings = next(read_tsv)
#print(csv_headings[0])
#print(len(csv_headings[0].split()))

i=0
max_itens = 200000

with open('dataset_msmarco/V04/msmarco_noise.csv', 'wt') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n', )
    writer.writerow(['label','title','text'])
    
    for row in read_tsv:
        #candidato base        
        ttitle = row[0]
        ttext = row[1]
        tnoise = row[2]
        
        #busca candidadato 2 e 3 negativos
        c_query = 0
        ttext2 = ""
        tnoise2 = ""
        tsv_file2 = open(f_main,encoding='utf-8')
        read_tsv2 = csv.reader(tsv_file2, delimiter="\t")
        for row2 in read_tsv2:
            if ttitle == row2[0] and row2[2]!=tnoise and row2[2]!=ttext2:
                if c_query == 0:
                    c_query = c_query + 1
                elif c_query == 1:
                    c_query = c_query + 1
                    ttext2 = row2[2]
                elif c_query == 2:
                    tnoise2 = row2[2]
                    break
        if ttext2 != "" and tnoise2 !="":
            
            #amostra positiva
            noise = tnoise.split()
            noise_n1 = noise[:len(noise)//2]
            noise_n2 = noise[len(noise)//2:]
            ttitle1 = 'text off: ' + " ".join(noise_n1) + ' text on: ' + ttitle + ' text off: ' + " ".join(noise_n2)
            ttext = 'text on: ' + ttext
            writer.writerow([1, ttitle1,ttext])
            #amostra negativa
            noise2 = tnoise2.split()
            noise2_n1 = noise2[:len(noise2)//2]
            noise2_n2 = noise2[len(noise2)//2:]
            ttitle2 = 'text off: ' + " ".join(noise2_n1) + ' text on: ' + ttitle + ' text off: ' + " ".join(noise2_n2)
            ttext2 = 'text on: ' + ttext2
            writer.writerow([0, ttitle2,ttext2])
            print('achou candidatos: ', i)
        else:
            print('nao achou candidatos')
        
        tsv_file2.close()

        if i> max_itens -2:
            print('chegou ao limite')
            break
        else:
            i = i +1
        print(i)

        
tsv_file.close()
