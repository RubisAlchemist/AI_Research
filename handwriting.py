# classification - handwriting dataset
# https://www.timeseriesclassification.com/description.php?Dataset=Handwriting

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import arff
from model import SiMBA, SiMBASimple, train_model_c

def load_arff_data(file_path):
    with open(file_path, 'r') as f:
        dataset = arff.load(f)
    data = np.array(dataset['data'])
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)
    y = y.astype(np.int64) - 1
    return X, y

class HandwritingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_handwriting(data_dir='./dataset/handwriting', batch=32):
    X1_test, y_test = load_arff_data(f'{data_dir}/HandwritingDimension1_TRAIN.arff')
    X2_test, _ = load_arff_data(f'{data_dir}/HandwritingDimension2_TRAIN.arff')
    X3_test, _ = load_arff_data(f'{data_dir}/HandwritingDimension3_TRAIN.arff')
    X1_train, y_train = load_arff_data(f'{data_dir}/HandwritingDimension1_TEST.arff')
    X2_train, _ = load_arff_data(f'{data_dir}/HandwritingDimension2_TEST.arff')
    X3_train, _ = load_arff_data(f'{data_dir}/HandwritingDimension3_TEST.arff')
    X_train = np.stack([X1_train, X2_train, X3_train], axis=-1)
    X_test = np.stack([X1_test, X2_test, X3_test], axis=-1)

    train_dataset = HandwritingDataset(X_train, y_train)
    test_dataset = HandwritingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    return train_loader, test_loader

def run_handwriting(batch=32, epochs=12, lr=3e-4, simple=False):
    train_loader, test_loader = load_handwriting(batch=batch)
    model = SiMBASimple(input_dim=3, num_classes=26) if simple else SiMBA(input_dim=3, num_classes=26)
    train_model_c(model, train_loader, test_loader, num_epochs=epochs, learning_rate=lr)