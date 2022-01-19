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


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, _ = self.gru(x, h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out


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
    hidden_size = 256
    num_layers = 2
    num_classes = 2
    dropout = 0.1
    batch_size = 512

    model = GRUModel(input_size, hidden_size, num_layers, num_classes, 0.2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Finally found the only good OOM-dataset
    dataset_train = (
        wds.WebDataset("shards/a/shard{0000000..000226}.tar")
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
        f1, pres, recall = check_other_metrics(test_loader, model, threshold=0.9)
        print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)},Train acc:, Validation acc:{acc}, pres:{pres}, recall:{recall}, F1: {f1}")
        torch.save(model, f'models/model{epoch}.pt')
