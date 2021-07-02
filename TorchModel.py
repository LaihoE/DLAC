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


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

def datacreator():
    y_train = []
    bigx = []
    X_train = []
    for folder in ["singlekills"]:
        for x in os.listdir(f"D:/Users/emill/csgocheaters/{folder}"):

            df = pd.read_csv(f"D:/Users/emill/csgocheaters/{folder}/{x}",index_col=0)
        #for cnt,x in enumerate(os.listdir(f"C:/Users/emill/PycharmProjects/open_anti_cheat/cleandrity/{folder}")):

            #df = pd.read_csv(f"C:/Users/emill/PycharmProjects/open_anti_cheat/cleandrity/{folder}/{x}",index_col=0)

            if len(df) > 20:
                df = df.select_dtypes(['number'])
                df = df.drop("sus",axis=1)   # junk
                #df = (df - df.mean()) / df.std()
                #df = df.fillna(df.mean)
                for i in range(20):
                    a = np.array(df.iloc[i])
                    X_train.append(a)
                bigx.append(X_train)
                if folder == 'singlekills':
                    y_train.append(1)
                elif folder == 'cleankills':
                    y_train.append(0)

    X_train = np.array(bigx)
    #y_train = np.array(y_train).astype(int)
    y_train = torch.tensor(y_train)
    y_train = y_train.type(torch.LongTensor)
    return X_train,y_train


class DemoDataset(Dataset):
    def __init__(self):
        X = []
        dirty = os.listdir(f"C:/Users/emill/PycharmProjects/open_anti_cheat/cleandrity/singlekills")
        clean = os.listdir(f"C:/Users/emill/PycharmProjects/open_anti_cheat/cleandrity/cleankills")
        for x in dirty:
            X.append((x, 1))
        for x in clean:
            X.append((x, 0))
        random.shuffle(X)
        print(X)
        self.data = X

    def __len__(self):
        return 200

    def __getitem__(self, index):
        file_info = self.data[index]
        if file_info[1] == 1:
            df = pd.read_csv(
                f"C:/Users/emill/PycharmProjects/open_anti_cheat/cleandrity/singlekills/{file_info[0]}",index_col=0)
            df = df.select_dtypes(['number'])
            df = df.drop("sus", axis=1)  # junk
            print(df)
            X = np.array(df.iloc[len(df) - 20:])
            X = torch.tensor(X)
            return X, file_info[1]

        elif file_info[1] == 0:
            df = pd.read_csv(
                f"C:/Users/emill/PycharmProjects/open_anti_cheat/cleandrity/singlekills/{file_info[0]}",index_col=0)
            df = df.select_dtypes(['number'])
            df = df.drop("sus", axis=1)  # junk
            print(df)
            X = np.array(df.iloc[len(df)-20:])
            X = torch.tensor(X)
            return X, file_info[1]



class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden



def train(train_loader, learn_rate, hidden_dim=256, EPOCHS=5, model_type="GRU"):
    # Setting common hyperparameters
    print(next(iter(train_loader))[0].shape)
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.clock()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.clock() - start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)
    print("sMAPE: {}%".format(sMAPE * 100))
    return outputs, targets, sMAPE

#############################################################################


dataset = DemoDataset()
first_data = dataset[0]
features, labels = first_data



batch_size = 2
#train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_data = dataset

dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True,pin_memory=True)
#data_iter = iter(dataloader)
#data = data_iter.next()

#features,labels = data
"""print(features)
lr = 0.001
gru_model = train(dataloader, lr, model_type="GRU")
"""

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
num_epochs = 100
batch_size = 2
learning_rate = 0.001

input_size = 15
sequence_length = 2
hidden_size = 64
num_layers = 3

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        images = images.to(device)
        labels = labels.to(device)

        print("images",images.dtype)
        print("labels",labels.dtype)


        # Forward pass
        outputs = model(images.float())

        labels=labels.long()

        loss = criterion(outputs.float(), labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')