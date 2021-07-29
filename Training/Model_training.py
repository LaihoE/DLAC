from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import os
from math import pi
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



class DemoDataset(Dataset):
    def __init__(self):
        files = os.listdir("c:/data/csgo/clean")
        self.files = [(i, i[0]) for i in files]

    def __getitem__(self, index):
        data = self.files[index]
        X = np.load(f"C:/data/csgo/clean/{data[0]}")
        if data[1] == "c":
            y = 0
        else:
            y = 1
        return X,y

    def __len__(self):
        return len(self.files)



class DemoDataset2(Dataset):
    def __init__(self):

        self.x = np.load("C:/Users/emill/PycharmProjects/CLIP/canmodel/x_tt.npy")
        self.y = np.load("C:/Users/emill/PycharmProjects/CLIP/canmodel/y_tt.npy")

        self.n_samples = len(self.y)
        print(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def __len__(self):
        return self.n_samples


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
            #print(scores)
            _, predictions = scores.max(1)
            print(scores)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()
    return float(num_correct) / float(num_samples) * 100



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


if __name__ == "__main__":
    scaler = MinMaxScaler()

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)


    dataset = DemoDataset()
    dataset_testing = DemoDataset2()

    num_classes = 2
    num_epochs = 30
    batch_size = 256
    learning_rate = 0.001

    input_size = 24
    sequence_length = 256
    hidden_size = 256
    num_layers = 2

    model = GRUModel(input_size, hidden_size, num_layers, num_classes,0.2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8
                                               )


    test_loader = torch.utils.data.DataLoader(dataset=dataset_testing,
                                               batch_size=batch_size,
                                               shuffle=True)
    n_total_steps = len(train_loader)

    eps = []
    losses = []

    accs=[]
    epocsl=[]
    acc_train = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            print(round(i*256 / 307000 *100,2),"%")
            images = images.to(device)
            labels = labels.to(device)
            print(labels)
            # Forward pass
            outputs = model(images.float())

            #labels=labels.long()
            print(outputs.shape)
            outputs = outputs.squeeze(0)
            sqs = torch.nn.Softmax(outputs)
            print(sqs)
            print(sqs.shape)
            loss = criterion(labels,outputs.float())

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

        #acc_tr = check_accuracy(train_loader, model)
        acc_train.append(acc)

        print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)},Train acc:NaN ,Validation acc:{acc}")
        torch.save(model, f"C:/Users/emill/PycharmProjects/CLIP/canmodel/grus/AntiCheat{epoch}.pt")
    torch.save(model, "AntiCheat3.pt")



    plt.plot(epocsl,accs,label = "line 1")
    plt.plot(epocsl,acc_train,label = "line 2")
    plt.show()