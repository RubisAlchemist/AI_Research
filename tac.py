# regressoin - TAC

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import SiMBA, SiMBASimple, train_model

class TACDataset(Dataset):
    def __init__(self, data_dir, sequence_length=1000):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.data = self._load_data()

    def _load_data(self):
        data_list = []
        pids = [f.split(' ')[0] for f in os.listdir(self.data_dir) if 'CAM Results.csv' in f]
        
        for pid in pids:
            cam_file = os.path.join(self.data_dir, f'{pid} CAM Results.csv')
            cam_data = pd.read_csv(cam_file)
            cam_data['Time'] = pd.to_datetime(cam_data['Time'])
            acc_file = os.path.join(self.data_dir, f'{pid}_acc_data.csv')
            acc_data = pd.read_csv(acc_file)
            acc_data['time'] = pd.to_datetime(acc_data['time'])
            acc_data = acc_data.sort_values('time').reset_index(drop=True)
            
            for idx, row in cam_data.iterrows():
                tac_time = row['Time']
                tac_value = row['TAC Level']
                
                acc_subset = acc_data[acc_data['time'] < tac_time].tail(self.sequence_length)
                if len(acc_subset) == self.sequence_length:
                    x = acc_subset[['x', 'y', 'z']].values
                    y = tac_value
                    data_list.append((x, y))
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

def load_tac(data_dir='./dataset/tac', batch=32, seq_len=1000):
    dataset = TACDataset(data_dir=data_dir, sequence_length=seq_len)
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=batch, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch, sampler=val_sampler)
    return train_loader, val_loader

def run_tac(seq_len=1000, batch=32, epochs=12, lr=3e-3, simple=False):
    train_loader, val_loader = load_tac(batch=batch, seq_len=seq_len)
    model = SiMBASimple(input_dim=3, num_classes=1) if simple else SiMBA(input_dim=3, num_classes=1)
    train_model(model, train_loader, val_loader, num_epochs=epochs, learning_rate=lr)