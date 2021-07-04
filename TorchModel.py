import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import os
import time
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)


def datacreator():
    y_train = []
    bigx = []

    for folder in ["singlekills","cleankills"]:
        for x in os.listdir(f"D:/Users/emill/csgocheaters/{folder}"):

            df = pd.read_csv(f"D:/Users/emill/csgocheaters/{folder}/{x}",index_col=0)

            #df = pd.read_csv(
                #f"C:/Users/emill/PycharmProjects/open_anti_cheat/cleandrity/singlekills/{file_info[0]}", index_col=0)
            df = df.select_dtypes(['number'])
            df = df.drop("sus", axis=1)  # junk

            df = df[["X", "Y", "Z", "ViewX", "ViewY"]]
            df[["X", "Y", "Z", "ViewX", "ViewY"]] = scaler.fit_transform(df[["X", "Y", "Z", "ViewX", "ViewY"]])
            #df[["ViewX", "ViewY"]] = scaler.fit_transform(df[["ViewX", "ViewY"]])



            if len(df) > 20:
                df = df.select_dtypes(['number'])
                #df = df.drop("sus",axis=1)   # junk

                X = np.array(df.iloc[len(df) - 100:])
                bigx.append(X)
                if folder == 'singlekills':
                    y_train.append(1)
                elif folder == 'cleankills':
                    y_train.append(0)

    X_train = np.array(bigx)
    return X_train,y_train

def datacreator_testing():
    y_train = []
    bigx = []

    for folder in ["singlekills","cleankills"]:
        for x in os.listdir(f"D:/Users/emill/csgotesting/{folder}"):

            df = pd.read_csv(f"D:/Users/emill/csgotesting/{folder}/{x}",index_col=0)

            #df = pd.read_csv(
                #f"C:/Users/emill/PycharmProjects/open_anti_cheat/cleandrity/singlekills/{file_info[0]}", index_col=0)
            df = df.select_dtypes(['number'])
            df = df.drop("sus", axis=1)  # junk

            df = df[["X","Y","Z","ViewX", "ViewY"]]
            df[["X","Y","Z","ViewX", "ViewY"]] = scaler.fit_transform(df[["X","Y","Z","ViewX", "ViewY"]])


            if len(df) > 20:
                df = df.select_dtypes(['number'])
                #df = df.drop("sus",axis=1)   # junk

                X = np.array(df.iloc[len(df) - 100:])
                bigx.append(X)
                if folder == 'singlekills':
                    y_train.append(1)
                elif folder == 'cleankills':
                    y_train.append(0)

    X_train = np.array(bigx)
    return X_train,y_train


class DemoDataset(Dataset):
    def __init__(self):

        x,y = datacreator()
        print(x,y)

        self.x = x
        self.y = y
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class DemoDataset2(Dataset):
    def __init__(self):

        x,y = datacreator_testing()
        print(x,y)

        self.x = x
        self.y = y
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples




dataset = DemoDataset()
dataset_testing = DemoDataset2()

batch_size=32
train_data = dataset

dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True,pin_memory=True)


dataloader_testing = DataLoader(dataset_testing, shuffle=True, batch_size=batch_size, drop_last=True,pin_memory=True)



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


num_classes = 2
num_epochs = 300
batch_size = 64
learning_rate = 0.0001

input_size = 5
sequence_length = 100
hidden_size = 256
num_layers = 10

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=dataset_testing,
                                           batch_size=batch_size,
                                           shuffle=True)
n_total_steps = len(train_loader)

eps = []
losses = []


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:


            x = x.to(device=device)
            y = y.to(device=device)

            x=x.float()
            y=y.long()

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()
    return float(num_correct) / float(num_samples) * 100

accs=[]
epocsl=[]

acc_train = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images.float())

        labels=labels.long()

        loss = criterion(outputs.float(), labels)
        losses.append(loss.item())
        eps.append(epoch)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
    acc="NA"
    acc = check_accuracy(test_loader,model)
    accs.append(acc)
    epocsl.append(epoch)

    acc_tr = check_accuracy(train_loader, model)
    acc_train.append(acc)

    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)},Train acc:{acc_tr} ,Validation acc:{acc}")

torch.save(model, "AntiCheat.pt")



plt.plot(epocsl,accs,label = "line 1")
plt.plot(epocsl,acc_train,label = "line 2")
plt.show()