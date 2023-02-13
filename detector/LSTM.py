from sklearn.preprocessing import minmax_scale
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from time import time
import json
import pandas as pd
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from .utils import apply_sliding_window, use_mini_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hiddent2out = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        lstm_out, _ = self.lstm(seq.view(self.batch_size, -1, self.input_dim))
        predict = self.hiddent2out(lstm_out)
        return predict[:, -1, :]


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
    predicted = predict(model, test_data, params)

    return predicted


def train(dataloader, input_dim, batch_size, n_epoch, lr=0.01):
    model = LSTM(input_dim, 100, batch_size).to(device)  #type: LSTM
    loss_function = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epoch):
        t0 = time()
        # print("epoch: %d / %d" % (epoch+1, n_epoch))

        loss_sum = 0
        for step, (batch_X, batch_Y) in enumerate(dataloader):
            model.zero_grad()
            predicted = model(batch_X)
            loss = loss_function(predicted.view(-1), batch_Y.view(-1))

            loss_sum += loss.item()
            if step % 100 == 0:
                # print(loss_sum / 100)
                loss_sum = 0
                # print(predicted - batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print("time: %.2f s" % float(time() - t0))
    return model


def predict(model, test_data, params):
    seq_len = params['seq_len']
    model.batch_size = 1
    predict_ls = []

    with torch.no_grad():
        for i in range(seq_len, len(test_data)):
            seq = torch.tensor(test_data[i-seq_len : i]).float().to(device)
            predicted = model(seq).item()
            predict_ls.append(predicted)

    padding = [.0] * (len(test_data) - len(predict_ls))

    return np.array(predict_ls + padding, dtype=np.float)

