import csv
import os
from numpy import loadtxt
import numpy as np
from numpy.lib.shape_base import column_stack
import pandas as pd

from datasets import load_dataset

os.system('clear')

demo = 0

if demo==0:
    f_main = '../../dataset_harvardnews/NewsArticles_clean_manual.csv'
    f_bm25 = '../../dataset_harvardnews/BM25_scores/HN_BM25_scores_manual.csv'
else:
    f_main = '../../dataset_harvardnews/NewsArticles_clean_demo.csv'
    f_bm25 = '../../dataset_harvardnews/BM25_scores/HN_BM25_scores_demo.csv'

#carrega dos arquivos csv
set_main = pd.read_csv(f_main, encoding='utf-8')
maxx= set_main.shape[0]
set_main_new = pd.DataFrame(columns=['label', 'title', 'text'])

bm25_aux = pd.read_csv(f_bm25, encoding='utf-8', names = ['index', 'article_id', 'bm25_max_index1', 'bm25_max_index2', 'bm25_max_index3'])

n_min = 450
n_max = 600
step_tag = 9999999 #a tag text on/off sera incluida a cada step

for index, row in set_main.iterrows():
    
    # negativo hard
    i1 = bm25_aux['bm25_max_index1'][index]
    i3 = bm25_aux['bm25_max_index3'][index]
    text1 = set_main['text'][i1]
    text_original = row['text']
    text1_noise = set_main['text'][i3]

    noise_size_start1 = np.random.randint(n_min,n_max)
    noise_size_end1 = np.random.randint(n_min,n_max)
    text_start1 = "text off: " + text1_noise[:noise_size_start1*5] + " text on: "
    text_end1 = " text off: " + text1_noise[-noise_size_end1*5:]

    #processa os textos
    text_start1_aux = text_start1.split()
    next_tag = step_tag + 2
    while len(text_start1_aux)-2 > next_tag:
        text_start1_aux = text_start1_aux[:next_tag] + ['text', 'off:'] + text_start1_aux[next_tag:]
        next_tag = next_tag + 2 + step_tag
    text_start1 = " ".join(text_start1_aux)

    text_end1_aux = text_end1.split()
    next_tag = step_tag + 2
    while len(text_end1_aux)-2 > next_tag:
        text_end1_aux = text_end1_aux[:next_tag] + ['text', 'off:'] + text_end1_aux[next_tag:]
        next_tag = next_tag + 2 + step_tag
    text_end1 = " ".join(text_end1_aux) 
    
    new_title1 = text_start1 + " " + row['title'] + " " + text_end1

    text1_aux = text1.split()
    next_tag = 0
    while len(text1_aux)-2 > next_tag:
        text1_aux = text1_aux[:next_tag] + ['text', 'on:'] + text1_aux[next_tag:]
        next_tag = next_tag + 2 + step_tag
    text1 = " ".join(text1_aux) 
    


    # noise hard
    i2= bm25_aux['bm25_max_index2'][index]
    text2 = set_main['text'][i2]

    noise_size_start = np.random.randint(n_min,n_max)
    noise_size_end = np.random.randint(n_min,n_max)
    text_start = "text off: " + text2[:noise_size_start*5] + " text on: "
    text_end = " text off: " + text2[-noise_size_end*5:]
    
    #processa os textos
    text_start_aux = text_start.split()
    next_tag = step_tag + 2
    while len(text_start_aux)-2 > next_tag:
        text_start_aux = text_start_aux[:next_tag] + ['text', 'off:'] + text_start_aux[next_tag:]
        next_tag = next_tag + 2 + step_tag
    text_start = " ".join(text_start_aux)

    text_end_aux = text_end.split()
    next_tag = step_tag + 2
    while len(text_end_aux)-2 > next_tag:
        text_end_aux = text_end_aux[:next_tag] + ['text', 'off:'] + text_end_aux[next_tag:]
        next_tag = next_tag + 2 + step_tag
    text_end = " ".join(text_end_aux)


    new_title = text_start + " " + row['title'] + " " + text_end

    text_original_aux = text_original.split()
    next_tag = 0
    while len(text_original_aux)-2 > next_tag:
        text_original_aux = text_original_aux[:next_tag] + ['text', 'on:'] + text_original_aux[next_tag:]
        next_tag = next_tag + 2 + step_tag
    text_original = " ".join(text_original_aux) 
    

    set_main_new = set_main_new.append(pd.DataFrame([['0', new_title1, text1]], columns=['label', 'title', 'text']), ignore_index=True)
    set_main_new = set_main_new.append(pd.DataFrame([['1', new_title, text_original]], columns=['label', 'title', 'text']), ignore_index=True)
    
    print(index)

if demo==0:
    set_main_new.to_csv('../../dataset_harvardnews/V03/NewsArticles_bm25_t3_600tk.csv', encoding='utf-8', index=False)
else:
    set_main_new.to_csv('../../dataset_harvardnews/V03/NewsArticles_bm25_t3_600tk_demo.csv', encoding='utf-8', index=False)
