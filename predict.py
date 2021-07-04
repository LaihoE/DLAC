import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        # or:
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
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

    def predict(self,x):
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
    df = pd.read_csv("Example_kill.csv")
    print(df)

    # Currently only using these 5
    df = df[["X", "Y", "Z", "ViewX", "ViewY"]]
    # Normalize
    df[["X", "Y", "Z", "ViewX", "ViewY"]] = scaler.fit_transform(df[["X", "Y", "Z", "ViewX", "ViewY"]])

    X = np.array(df.iloc[len(df) - 100:])
    X = X.reshape((1,100,5))
    X = torch.tensor(X)
    X = X.to(device)
    X = X.float()

    model = torch.load(r"C:\Users\emill\PycharmProjects\CLIP\canmodel/AntiCheat.pt")
    print(model.predict(X))