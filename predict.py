import os
from math import pi
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time


scaler = MinMaxScaler()


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out

    def predict(self,x):
        model.eval()
        with torch.no_grad():
            # Set initial hidden states (and cell states for LSTM)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

            # x: (n, 28, 28), h0: (2, n, 128)

            # Forward propagate RNN
            out, _ = self.rnn(x, h0)
            # or:
            # out, _ = self.lstm(x, (h0,c0))

            # out: tensor of shape (batch_size, seq_length, hidden_size)
            # out: (n, 28, 128)

            # Decode the hidden state of the last time step
            out = out[:, -1, :]
            # out: (n, 128)

            out = self.fc(out)
            # out: (n, 10)
            return out


if __name__ == "__main__":
    X = np.load("prd.npy")
    print(X.shape)
    X = torch.tensor(X)
    X = X.to(device)
    X = X.float()

    model = torch.load(r"C:\Users\emill\PycharmProjects\CLIP\canmodel/AntiCheat.pt",map_location=torch.device('cpu'))
    prd = model.forward(X)
    probs = torch.softmax(prd,1)
    for i in range(500):
        print("Probability of cheating:",round(1-probs[i][0].item(),2))