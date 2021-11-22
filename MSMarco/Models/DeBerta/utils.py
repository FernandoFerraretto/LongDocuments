import torch
import numpy as np
from urllib.parse import unquote
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
import gc

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, mydata, seq_length, tokenizer):
        self.labels = np.array(mydata['label'])
        self.seq1 = np.array(mydata['title'])
        self.seq2 = np.array(mydata['text'])
        self.delta_min = 20 #numero de tokens minimos na segunda sequencia
        self.seq_length = seq_length - self.delta_min
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        seq1 = self.seq1[idx]
        seq2 = self.seq2[idx]
        
        item1= self.tokenizer(seq1, truncation=True, max_length= self.seq_length)
        mxx= self.seq_length-len(item1['input_ids'])+1+self.delta_min
        if mxx<self.delta_min: mxx=self.delta_min
        item2= self.tokenizer(seq2, truncation=True, max_length=mxx, padding='max_length')

        item={'input_ids': torch.tensor(item1['input_ids']+item2['input_ids'][1:])}
        item['attention_mask']=torch.tensor(item1['attention_mask']+item2['attention_mask'][1:])
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

    def __len__(self):
        return len(self.labels)

class MyModel(pl.LightningModule):

    def __init__(self,model,device):
        super(MyModel, self).__init__()
        self.model = model
        self.mydevice = device
        
    def training_step(self, batch, batch_nb):
        input_ids = batch['input_ids'].to(self.mydevice)
        attention_mask = batch['attention_mask'].to(self.mydevice)
        labels = batch['label'].to(self.mydevice)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        
        #loss
        loss = outputs[0]

        #acc
        outputs = outputs['logits']
        out_labels = torch.argmax(outputs,axis=1)
        comp = (labels==out_labels)
        corretos = comp.sum()
        num_cases = len(labels)
        acc = (corretos/num_cases)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        input_ids = batch['input_ids'].to(self.mydevice)
        attention_mask = batch['attention_mask'].to(self.mydevice)
        labels = batch['label'].to(self.mydevice)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        
        #loss
        loss = outputs[0]

        #acc
        outputs = outputs['logits']
        out_labels = torch.argmax(outputs,axis=1)
        comp = (labels==out_labels)
        corretos = comp.sum()
        num_cases = len(labels)
        acc = (corretos/num_cases)
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        self.model.train()
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.01)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.25),
            'name': 'log_lr'
            }
        return [optimizer], [lr_scheduler]