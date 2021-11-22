#inicio / imports
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import TrainingArguments, AdamW
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
import wget
import os.path
from os import path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor

neptune_logger = NeptuneLogger(
            api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMjU2OTI2MS01YzQ0LTRjNTItODdhNC05ODdhOThlNTg4OTIifQ==',
            project_name='feferraretto/Model-AAN', 
        )

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

#os.system('clear')

# se 1: utiliza a amostra de demonstracao
# se 0: utiliza o dataset real
demo = 0
verbose = 1


#-----------------------------------------------------------------------------------------------------------------
#dataset
#-----------------------------------------------------------------------------------------------------------------
from transformers import BigBirdTokenizerFast, BigBirdForSequenceClassification
tokenizer = BigBirdTokenizerFast.from_pretrained('google/bigbird-roberta-base')
#tokenizer = BigBirdTokenizerFast.from_pretrained('google/bigbird-roberta-large')


#define o paths
f_main = '../../dataset_msmarco/V04/msmarco_longnoise700.csv'


#carrega dos arquivos csv
if demo ==0:
    #set_train = load_dataset('csv', delimiter = ',', data_files=f_main, split='train[:20000]')
    #set_eval = load_dataset('csv', delimiter = ',', data_files=f_main, split='train[20000:]')
    
    set_train = load_dataset('csv', delimiter = ',', data_files=f_main, split='train[:80000]')
    set_eval = load_dataset('csv', delimiter = ',', data_files=f_main, split='train[80000:]')
    
        
else:
    set_train = load_dataset('csv', delimiter = ',', data_files=f_main, split='train[:100]')
    set_eval = load_dataset('csv', delimiter = ',', data_files=f_main, split='train[100:120]')

#shape do dataset
print()
print('Train:', set_train.shape)
print('Eval:', set_eval.shape)
print()

#criando os datasets
from utils import MyDataset

#-- nonoise
#max_length = 320
#-- noise250
#max_length = 1024 
#-- noise450
#max_length = 1512
#-- noise700
max_length = 2432 

batch_n = 1

#cria datasets
train_dataset = MyDataset(set_train,max_length, tokenizer)
eval_dataset = MyDataset(set_eval,max_length, tokenizer)

#testa um item do dataset - para visualização
if verbose == 1:
    print()
    idx = 2
    x1= train_dataset.__getitem__(idx)
    print()
    print('       itens dataset: ',train_dataset.__len__())
    print('     input_ids shape: ', x1['input_ids'].shape)
    print('Attention_mask shape: ', x1['attention_mask'].shape)
    print('               Label:', x1['label'])
    print('\nDecodifica uma amosta:\n', tokenizer.decode(x1['input_ids']))
    print()


#cria o data loader de test
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_n, num_workers = 0, shuffle=True)
eval_dataloader =  torch.utils.data.DataLoader(eval_dataset, batch_size=batch_n, num_workers = 0)

#modelo
from utils import MyModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if verbose == 1:
    print()
    print('RODANDO COM CUDA: ', torch.cuda.is_available() )
    print()

num_gpus = 0
if torch.cuda.is_available():
    num_gpus = 1

model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base')
#model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-large')

model.to(device)
model.train()


#treinamento
num_epoch = 20
accum_batch = 32

num_batch = int(np.ceil(train_dataset.__len__() / batch_n))
model_pl = MyModel(model, device)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(accumulate_grad_batches=accum_batch, gpus=num_gpus, max_epochs=num_epoch, logger=neptune_logger, callbacks=[lr_monitor])
trainer.fit(model_pl,train_dataloader,eval_dataloader)


