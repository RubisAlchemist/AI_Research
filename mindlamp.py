# regression, classification - mindlamp

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import SiMBA, SiMBASimple, train_model, train_model_c

class LAMPDataset(Dataset):
    def __init__(self, data_dir, y_path, seq_len, target, is_regression):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.y_data = pd.read_csv(y_path)
        self.uuids = self.y_data['uuid'].values
        self.scores = self.y_data[target].values
        self.scaler = StandardScaler()
        self.is_regression = is_regression

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        score = self.scores[idx]
        file_path = os.path.join(self.data_dir, f'{uuid}.csv')
        data = pd.read_csv(file_path)
        data = data.sort_values(by='timestamp').reset_index(drop=True)
        time_series_data = data.drop(columns=['timestamp']).values
        time_series_data = self.scaler.fit_transform(time_series_data)
        if time_series_data.shape[0] < self.seq_len:
            padding = np.zeros((self.seq_len - time_series_data.shape[0], time_series_data.shape[1]))
            time_series_data = np.vstack((time_series_data, padding))
        elif time_series_data.shape[0] > self.seq_len:
            time_series_data = time_series_data[:self.seq_len, :]
        if self.is_regression:
            return torch.tensor(time_series_data, dtype=torch.float32), torch.tensor(score, dtype=torch.float32)
        else:
            return torch.tensor(time_series_data, dtype=torch.float32), torch.tensor(score, dtype=torch.int64)

def load_lamp(target, batch_size, seq_len, is_regression, data_dir='./dataset/lamp', y_file='./dataset/lamp/student_info.csv'):
    dataset = LAMPDataset(data_dir=data_dir, y_path=y_file, seq_len=seq_len, target=target, is_regression=is_regression)
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    return train_loader, val_loader

def run_regression(target, batch=8, seq_len=6000, epochs=12, lr=3e-3, simple=False):
    train_loader, val_loader = load_lamp(target=target, batch_size=batch, seq_len=seq_len, is_regression=True)
    model = SiMBASimple(input_dim=15, num_classes=1, scale=80) if simple else SiMBA(input_dim=15, num_classes=1, scale=1)
    train_model(model, train_loader, val_loader, num_epochs=epochs, learning_rate=lr)

def run_classification(target, batch=8, seq_len=6000, epochs=12, lr=3e-3, simple=False):
    train_loader, val_loader = load_lamp(target=target, batch_size=batch, seq_len=seq_len, is_regression=False)
    model = SiMBASimple(input_dim=15, num_classes=2, scale=1) if simple else SiMBA(input_dim=15, num_classes=2, scale=1)
    train_model_c(model, train_loader, val_loader, num_epochs=epochs, learning_rate=lr)