# Model

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from simba import Block_mamba, ClassBlock
from timm.models.layers import trunc_normal_


'''
Simple SiMBA model
x -> embedding layer -> SiMBA blocks -> output layer -> output
'''
class SiMBASimple(nn.Module):
    def __init__(self,
        num_classes, 
        input_dim, 
        embed_dim=64, 
        mlp_ratio=4, 
        num_blocks=4,
        scale=1,
    ):
        super().__init__()
        self.scale = scale
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList([
            Block_mamba(dim=embed_dim, mlp_ratio=mlp_ratio, cm_type="EinFFT")
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x): # x : B x L x D
        x = self.embedding(x)
        B, L, D = x.shape
        for block in self.blocks:
            x = block(x, L, D)
        x = self.norm(x.mean(dim=1))
        x = self.head(x) * self.scale
        return x

'''
Model based on SiMBA image classification model architecture 
https://github.com/badripatro/simba
'''
class SiMBA(nn.Module):
    def __init__(self,
        input_dim,
        num_classes,
        embed_dims=[64, 128, 320, 448],
        mlp_ratios=[8, 8, 4, 4], 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3], 
        sr_ratios=[4, 2, 1, 1], 
        num_stages=4,
        scale=1,
        **kwargs
    ):
        super().__init__()
        self.scale = scale
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        alpha=5
        for i in range(num_stages):
            if i == 0:
                patch_embed = nn.Linear(input_dim, embed_dims[i])
            else:
                patch_embed = nn.Linear(embed_dims[i - 1], embed_dims[i])

            block = nn.ModuleList([Block_mamba(
                dim=embed_dims[i], 
                mlp_ratio = mlp_ratios[i], 
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer,
                sr_ratio = sr_ratios[i],
                cm_type='EinFFT')
            for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        post_layers = ['ca']
        self.post_network = nn.ModuleList([
            ClassBlock(
                dim = embed_dims[-1], 
                mlp_ratio = mlp_ratios[-1],
                norm_layer=norm_layer,
                cm_type='EinFFT')
            for _ in range(len(post_layers))
        ])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_cls(self, x, H, W):
        B, N, C = x.shape
        cls_tokens = x.mean(dim=1, keepdim=True)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x, H, W)
        return x

    def forward(self, x):
        B, L, D = x.shape
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x = patch_embed(x)
            for blk in block:
                x = blk(x, L, D)
            
            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                #x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.forward_cls(x, L, D)[:, 0]
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)
        x = self.head(x) * self.scale
        return x

# train code

# 1. train codes for regression model
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            #print(outputs, targets)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(val_loader.dataset)
    return epoch_loss

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device='cuda'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')
    print('Training complete.')

# 2. train codes for classification model

def train_one_epoch_c(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

def validate_one_epoch_c(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

def train_model_c(model, train_loader, val_loader, num_epochs, learning_rate, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss, train_acc = train_one_epoch_c(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch_c(model, val_loader, criterion, device)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')
    print('Training complete.')