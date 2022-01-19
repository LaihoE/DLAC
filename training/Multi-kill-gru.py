import joblib
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from sklearn.metrics import f1_score,precision_score,recall_score
import webdataset as wds


class TestDataset(Dataset):
    def __init__(self):
        scaler = joblib.load('scaler.gz')
        print(f"Scaler fitted on {scaler.n_samples_seen_} samples",)
        self.x = np.float32(np.load("F:/csgo/transformer_data/test/X.npy"))
        self.x = self.x.reshape(-1, 5)
        self.x = scaler.transform(self.x)
        self.x = self.x.reshape(-1, 128, 5)

        self.y = np.load("F:/csgo/transformer_data/test/y.npy")
        self.y = np.int64(self.y)
        self.n_samples = len(self.y)
        print(np.where(self.y == 0))
        print(np.where(self.y == 1))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def check_accuracy(loader, model):
    num_correct, num_samples = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.float()
            y = y.long()
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()
    return float(num_correct) / float(num_samples) * 100


def check_other_metrics(loader, model, threshold):
    model.eval()
    F1 = []
    pres = []
    recall = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.float()
            y = y.long()
            scores = model(x)
            _, predictions = scores.max(1)
            probs = torch.softmax(scores, 1)
            y = y.cpu()
            probs = probs.detach().cpu()
            probs = probs[:,1] > threshold
            F1.append(f1_score(y,probs))
            pres.append(precision_score(y,probs))
            recall.append(recall_score(y,probs))
    model.train()
    return sum(F1) / len(F1), sum(pres) / len(pres), sum(recall) / len(recall)


class GruCeption(nn.Module):
    def __init__(self, input_dim, first_hidden_dim, second_hidden_dim, layer_dim, output_dim, dropout_prob, number_of_kills):
        super(GruCeption, self).__init__()
        self.number_of_kills = number_of_kills
        self.layer_dim = layer_dim
        self.hidden_dim = first_hidden_dim
        self.gru = nn.GRU(input_dim, first_hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(second_hidden_dim, output_dim)
        self.gru2 = nn.GRU(number_of_kills, second_hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)

    def forward(self, X):
        print(X.shape)
        # first kill like this for easy torch.cat
        x = X[:,0,:,:]
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, _ = self.gru(x, h0.detach())
        out1 = out[:, -1, :]
        o = out1.reshape((-1, 1, self.hidden_dim))
        # Loop over kills 1->n kills
        for i in range(1, self.number_of_kills):
            x = X[:, i, :, :]
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
            out, _ = self.gru(x, h0.detach())
            out2 = out[:, -1, :]
            out2 = out2.reshape((-1, 1, self.hidden_dim))
            o = torch.cat((o, out2), dim=1)

        # Batch nkills hsize -> Batch hsize nkills
        o = rearrange(o,'b n h -> b h n')
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, _ = self.gru2(o, h0.detach())
        out = out[:, -1, :]
        return self.fc(out)


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    learning_rate = 0.002
    input_size = 5
    embedding_dim = 5
    first_hidden_size = 256
    second_hidden_size = 256
    num_layers = 1
    num_classes = 2
    dropout = 0.1
    batch_size = 256

    model = GruCeption(input_size, first_hidden_size, second_hidden_size, num_layers, num_classes, 0.2, number_of_kills=25).to(device)
    #model = GruCeption(input_size, first_hidden_size, second_hidden_size, num_layers, num_classes, 0.2, number_of_kills=25).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Finally found the only good OOM-dataset
    dataset_train = (
        wds.WebDataset("shards/a/shard{0000000..000227}.tar")
        .shuffle(1000)
        .decode()
        .to_tuple("x.pyd", "y.cls")
    )
    dataset_testing = TestDataset()

    train_loader = torch.utils.data.DataLoader(dataset_train, num_workers=12, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_testing,
                                              batch_size=batch_size,
                                              shuffle=True)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(100):
        losses = []
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, labels)

            losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # mid epoch check
            if i % 1000 == 0:
                acc = check_accuracy(test_loader, model)
                f1, pres, recall = check_other_metrics(test_loader, model, threshold=0.9)
                print(f"Cost at epoch {i*batch_size} is {sum(losses) / len(losses)},Train acc:, Validation acc:{acc}, pres:{pres}, recall:{recall}, F1: {f1}")
                torch.save(model, f'models/model{i*batch_size}.pt')
        # Normal check
        acc = check_accuracy(test_loader, model)
        f1,pres,recall = check_other_metrics(test_loader, model, threshold=0.9)
        print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)},Train acc:, Validation acc:{acc}, pres:{pres}, recall:{recall}, F1: {f1}")
        torch.save(model, f'models/model{epoch}.pt')
