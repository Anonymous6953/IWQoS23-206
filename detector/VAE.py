import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import json

from .utils import use_mini_batch, apply_sliding_window

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(self, sel_len=100, h_dim=500, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(sel_len, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)  # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim)  # 保准方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, sel_len)

    # 编码过程
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码过程
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    # 整个前向传播过程：编码->解码
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


def train(dataloader, input_dim, batch_size, n_epoch, lr=0.01):

    model = VAE(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epoch):
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            
            # 获取样本，并前向传播
            x_reconst, mu, log_var = model(x)

            # 计算重构损失和KL散度（KL散度用于衡量两种分布的相似程度）
            # KL散度的计算可以参考论文或者文章开头的链接
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # 反向传播和优化
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (i + 1) % 10 == 0:
            #     print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
            #           .format(epoch + 1, n_epoch, i + 1, len(train_data_loader), reconst_loss.item(), kl_div.item()))

    return model


def reconstruct(model, test_data, params):
    seq_len = params['seq_len']
    reconstruct_ls = []

    with torch.no_grad():
        for i in range(seq_len, len(test_data)):
            seq = torch.tensor(test_data[i-seq_len : i]).float().to(device)
            predicted = model(seq)[0]
            reconstruct_ls.append(predicted[0].item())
    padding = [0] * (len(test_data) - len(reconstruct_ls))
    return np.array(reconstruct_ls + padding)


def train_and_predict(train_seq, test_seq, params):
    seq_len = params['seq_len']
    train_data = train_seq.to_numpy()
    test_data = test_seq.to_numpy()
    batch_size = params['batch_size']
    lr = params['lr']
    epochs = params['epochs']

    seq_dataset, seq_ground_truth = apply_sliding_window(train_data, seq_len=seq_len, flatten=False)
    train_data_loader = use_mini_batch(seq_dataset, seq_ground_truth, batch_size, device=device)

    input_dim = train_data_loader.dataset.feature_len

    model = train(train_data_loader, input_dim, batch_size, epochs, lr)
#     predicted = reconstruct(model, test_data, params)

#     return predicted

    return np.zeors(100)
