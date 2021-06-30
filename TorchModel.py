import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import os
import time

import torch
import torch.nn as nn


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



def datacreator():
    y_train = []
    bigx = []
    X_train = []
    for folder in ["singlekills"]:
        for cnt,x in enumerate(os.listdir(f"D:/Users/emill/csgocheaters/{folder}")):

            df = pd.read_csv(f"D:/Users/emill/csgocheaters/{folder}/{x}",index_col=0)

            if len(df) > 20:
                df = df.select_dtypes(['number'])
                df = df.drop("sus",axis=1)   # junk
                #df = (df - df.mean()) / df.std()
                #df = df.fillna(df.mean)
                print(df.isna().sum())
                for i in range(20):
                    a = np.array(df.iloc[i])
                    X_train.append(a)
                bigx.append(X_train)
                if folder == 'singlekills':
                    y_train.append(1)
                elif folder == 'cleankills':
                    y_train.append(0)


    X_train = np.array(bigx)

    return X_train,y_train





class DemoDataset(Dataset):
    def __init__(self):
        bigboi = pd.read_csv(r"D:\Users\emill\csgocheaters\singlekills/1.csv")
        for x in os.listdir(f"D:/Users/emill/csgocheaters/singlekills"):
            small_df = pd.read_csv(f"D:/Users/emill/csgocheaters/singlekills/{x}", index_col=0)
            bigboi = pd.concat([bigboi, small_df], ignore_index=True)

        print(bigboi)

        df = bigboi

        #df = pd.read_csv(r"D:\Users\emill\csgocheaters\singlekills/11.csv")
        boolcols = ["HasDefuse", "HasHelmet", "IsAirborne", "IsAlive", "IsDucking", "IsFlashed", "IsScoped",
                    "IsWalking"]
        for x in boolcols:
            df[x] = df[x].astype(int)
        # Don't need the players name
        df = df.drop("Name", axis=1)

        # Could be used
        catcolumns = ["ActiveWeapon", "AreaName","Inventory"]
        for x in catcolumns:
            df = df.drop(x, axis=1)
        print(df.dtypes)


        xy = df.to_numpy()

        xy = np.load(r"D:\Users\emill\csgocheaters\singlekills/11.csv")
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(np.ones(1))
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples



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




dataset = DemoDataset()
first_data = dataset[0]
features, labels = first_data
print(features,labels)
print(features.shape)

batch_size = 2
#train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_data = dataset
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
lr = 0.001
gru_model = train(train_loader, lr, model_type="GRU")

# Under progress