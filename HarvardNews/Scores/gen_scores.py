#inicio / imports
import os
import torch
import numpy as np
from numpy import savetxt
from torch.utils.data import DataLoader
from transformers import TrainingArguments, AdamW
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset
from datasets import load_dataset
import wget
import os.path
from os import path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from gensim.summarization.bm25 import BM25 # type: ignore
import scipy.stats as ss
import matplotlib.pyplot as plt



os.system('clear')

# se 1: utiliza a amostra de demonstracao
# se 0: utiliza o dataset real
demo = 0
verbose = 1


#-----------------------------------------------------------------------------------------------------------------
#dataset
#-----------------------------------------------------------------------------------------------------------------
f_main = 'dataset_harvardnews/NewsArticles_clean_manual.csv'

if demo==1:
    f_main = 'dataset_harvardnews/NewsArticles_clean_manual_demo.csv'

    
    
#carrega dos arquivos csv
set_main = load_dataset('csv', data_files=f_main)

#shape do dataset
print()
print('Shape Dataset:', set_main.shape)
print()

#criando o vetor
corpus = set_main['train']['text']

def simple_tok(sent:str):
     return sent.split()

tok_corpus = [simple_tok(s) for s in corpus]
bm25 = BM25(tok_corpus)

if verbose == 1:
    print('Quantidade de sequencias no corpus: ', bm25.corpus_size)
    print('Quantidade de tokens no corpus: ', len(bm25.idf.keys()))
    print()


#calcula scores
index=0

bm25_data = np.array([])

for item in set_main['train']:
    query = item['text'].split()
    bm_scores = bm25.get_scores(query)
    i_max = np.argsort(np.array(bm_scores))[-2]
    i_max2 = np.argsort(np.array(bm_scores))[-3]
    i_max3 = np.argsort(np.array(bm_scores))[-4]
    
    xitem = np.array([[index,item['article_id'],i_max, i_max2, i_max3]])
    

    if len(bm25_data)>0:
        bm25_data= np.append(bm25_data, xitem, axis=0)

    
    else:
        bm25_data= xitem

    index = index+1
    print('--- ', index)


#salva a saida
#columns: index, article_id, bm25_max_index1, bm25_max_index2 
savetxt('dataset_harvardnews/BM25_scores/HN_BM25_scores_manual.csv', bm25_data, delimiter=',')

if verbose == 1:
    print(bm25_data)
    print(bm25_data.shape)
    print()
