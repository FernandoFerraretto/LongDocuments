import os
from numpy import loadtxt
import numpy as np
from numpy import savetxt
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification



os.system('clear')

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

f_main = '../dataset_harvardnews/V03/NewsArticles_bm25_t3_600tk.csv'

#carrega dos arquivos csv
set_main = load_dataset('csv', delimiter = ',', data_files=f_main)

seq1 = set_main['train']['title']
seq2 = set_main['train']['text']
data_len = np.array([])
    

for item in set_main['train']:

    x1 = tokenizer(item['title'])
    x2 = tokenizer(item['text'])
    l1 = len(x1['input_ids'])
    l2 = len(x2['input_ids'])
    lt = l1+l2
    if len(data_len)>0:
        data_len = np.append(data_len,np.array([[l1,l2, lt]]),axis=0)
    else:
        data_len = np.array([[l1,l2, lt]])


savetxt('../dataset_harvardnews/data_len_bm25.csv', data_len, delimiter=',')
