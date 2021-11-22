import os
import numpy as np
from numpy import savetxt
import random

os.system('clear')

noise_min_size = 0

import csv
f_main = 'dataset_msmarco/V04/msmarco_noise.csv'
csv_file = open(f_main,encoding='utf-8')
read_csv = csv.reader(csv_file, delimiter=",")
next(read_csv)

count = 0
with open('dataset_msmarco/V04/msmarco_nonoise.csv', 'wt') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n', )
    writer.writerow(['label','title','text'])
    
    for row in read_csv:
        #candidato base        
        tlabel = row[0]
        seq1 = row[1]
        seq2 = row[2]
        
        seq1 = seq1.split()
    
        
        #retrieve noise
        noise = []
        stage= 0
        for i in range(len(seq1)):
            if seq1[i-2] == "text" and seq1[i-1] == "off:":
                stage = 1
            elif seq1[i] == "text" and seq1[i+1] == "on:":
                stage = 0
            if stage ==1: noise.append(seq1[i])
        
        #increase noise
        noise_increase1 = []
        while len(noise_increase1) < noise_min_size:
            ini = random.randint(0,len(noise)//2)
            end = random.randint(len(noise)//2,len(noise))
            noise_increase1= noise_increase1 + noise[ini:end]
        noise_increase2 = []
        while len(noise_increase2) < noise_min_size:
            ini = random.randint(0,len(noise)//2)
            end = random.randint(len(noise)//2,len(noise))
            noise_increase2= noise_increase2 + noise[ini:end]


        #retrieve title
        ttitle = []
        stage= 0
        for i in range(len(seq1)):
            if seq1[i-2] == "text" and seq1[i-1] == "on:":
                stage = 1
            elif seq1[i] == "text" and seq1[i+1] == "off:":
                stage = 0
            if stage ==1: ttitle.append(seq1[i])
        
        seq1 = 'text off: ' + " ".join(noise_increase1) + ' text on: ' + " ".join(ttitle) + ' text off: ' + " ".join(noise_increase2)
        writer.writerow([tlabel, seq1,seq2])
        print(count)
        count = count + 1


csv_file.close()



