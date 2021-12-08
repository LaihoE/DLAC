import pandas as pd
import torch
import torch.nn as nn
import subprocess as sp
import math
import numpy as np
import csv
import copy

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using: {device}")
device = 'cpu'


class BiGRU(nn.Module):
    """A pyTorch Bi-Directional LSTM RNN implementation"""

    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, batch_size, dropout, device):
        super(BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.device = device

    def _init_hidden(self, current_batch_size):
        h0 = torch.zeros(self.num_layers * 2, current_batch_size, self.hidden_dim)#.to(self.device)
        c0 = torch.zeros(self.num_layers * 2, current_batch_size, self.hidden_dim)#.to(self.device)
        return h0, c0

    def forward(self, x):
        h, c = self._init_hidden(current_batch_size=x.size(0))
        out, _ = self.lstm(x, (h, c))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.gru(x, h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class Model:
    def __init__(self,dem_folder,raw=False):
        self.raw = raw
        pipe = sp.Popen(f'go run parser.go {dem_folder}', shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        res = pipe.communicate()
        for line in res[0].decode(encoding='utf-8').split('\n'):
            print(line)
        self.X = pd.read_csv('data/data.csv', header=None).to_numpy()
        self.model = torch.load('best.pt', map_location='cpu')
        self.model.eval()

    def predict(self):
        X = self.X
        X = X.reshape(-1,128,9)
        batch_size = 5000
        file = X[:, :, 0]
        name = X[:, :, 1]
        id = X[:,:,2]
        tick = X[:,:,3]
        X = X[:,:,4:]
        X = np.float32(X)
        timeline = []
        total_batches = math.ceil(X.shape[0] / batch_size)
        sussies = []
        detection_dict = {}
        all_shots_dict = {}
        # Headers
        with open('output.csv', 'w', newline='\n')as f:
            thewriter = csv.writer(f)
            thewriter.writerow(["SteamId", "Total_shots", "Detections", "Detections/Total_shots"])
        for inx in range(total_batches):
            print(inx * batch_size)
            x = X[inx * batch_size:inx * batch_size + batch_size, :, :]
            ids = id[inx * batch_size:inx * batch_size + batch_size, :]
            ticks = tick[inx * batch_size:inx * batch_size + batch_size, :]
            names = name[inx * batch_size:inx * batch_size + batch_size]
            files = file[inx * batch_size:inx * batch_size + batch_size]
            og = copy.deepcopy(x)
            for i in range(total_batches):
                x = og[i * batch_size:i * batch_size + batch_size]
                x = torch.tensor(x).to(device).float()
                prd = self.model(x)
                probs = torch.softmax(prd, 1)
                for shot in range(x.shape[0]):
                    probability = probs[shot][1].item()
                    sussies.append((names[shot][0], round(probability, 2) * 100))
                    timeline.append((names[shot][0], ticks[shot][0], round(probability, 2)))
                    if ids[shot][0] not in all_shots_dict:
                        all_shots_dict[ids[shot][0]] = 1
                    else:
                        all_shots_dict[ids[shot][0]] += 1
                    if probability > 0.95 and ids[shot][0] != 0:
                        if ids[shot][0] not in detection_dict:
                            detection_dict[ids[shot][0]] = 1
                        else:
                            detection_dict[ids[shot][0]] += 1
                        # print("Name:",names[shot][0],"SteamId:",ids[shot][0],"Cheating:", round(probability, 2)*100, "%         ","Tick:",ticks[shot][0],"File:",files[shot][0])
                        if self.raw:
                            with open('cheaters.csv', 'a', newline='\n', encoding='UTF-8')as f:
                                thewriter = csv.writer(f)
                                thewriter.writerow(
                                    [names[shot][0], ids[shot][0], round(probability, 2) * 100, ticks[shot][0],
                                     files[shot][0]])
        preddict = {}
        for k, v in detection_dict.items():
            preddict[k] = [all_shots_dict[k], detection_dict[k], detection_dict[k] / all_shots_dict[k]]

        # Outputs come from here
        out = {k: v for k, v in sorted(preddict.items(), key=lambda item: item[1][2], reverse=True)}
        with open('output.csv', 'a', newline='\n')as f:
            thewriter = csv.writer(f)
            for k, v in out.items():
                print(k, v[0], v[1], v[2])
                thewriter.writerow([k, v[0], v[1], v[2]])


if __name__ == "__main__":
    demo_folder = "./demo/"
    model = Model(demo_folder)
    model.predict()