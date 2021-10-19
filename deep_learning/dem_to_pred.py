import pandas as pd
import torch
import torch.nn as nn
import subprocess as sp
import math
import numpy as np
import joblib
import csv


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using: {device}")


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


class Model:
    def __init__(self,df_path):
        self.df = pd.read_csv(df_path,header=None)
        self.X = None


    def extract_features(self):
        num_samples = len(self.df)
        data = self.df.to_numpy()
        X = data

        features = [x for x in range(9)]
        num_features = len(features)
        idxs = []
        for i in range(X.shape[1]):
            if (i >= num_features * num_samples):
                continue
            if (i % num_features) in features:
                idxs.append(i)

        X = X.reshape(-1, 128, num_features)
        return X

    def predict(self,batch_size):

        X = self.extract_features()
        file = X[:,:,0]
        name = X[:,:,1]
        id = X[:,:,2]
        tick = X[:,:,3]
        X = X[:,:,4:]

        print(f"Shape: {X.shape}")
        X = X.astype(np.float32)
        # Deep learning model. Remove map_location if GPU-enabled PyTorch. Enabling GPU speeds up predictions, but may
        # not be needed if predicting small amounts of games
        model = torch.load("Anticheat2.pt",map_location=torch.device('cpu'))

        for parameter in model.parameters():
            print(parameter)

        # rounded up n shots / batch size
        total_batches = math.ceil(X.shape[0] / batch_size)
        counter = 0
        sussies = []
        with open('cheaters.csv', 'w', newline='\n')as f:
            thewriter = csv.writer(f)
            thewriter.writerow(["Name","SteamId","Cheating%","Tick","File"])
        for inx in range(total_batches):
            print(inx*batch_size)
            x = X[inx * batch_size:inx * batch_size + batch_size, :, :]
            ids = id[inx * batch_size:inx * batch_size + batch_size,:]
            ticks = tick[inx * batch_size:inx * batch_size + batch_size,:]
            names = name[inx * batch_size:inx * batch_size + batch_size]
            files = file[inx * batch_size:inx * batch_size + batch_size]

            x = torch.tensor(x).to(device).float()
            prd = model.forward(x)
            probs = torch.softmax(prd, 1)
            # Each shot in batch
            for shot in range(x.shape[0]):
                probability = probs[shot][1].item()

                if probability > 0.95:                                # sampled 64-tick so need to *2 if 128 server
                    print("Name:",names[shot][0],"SteamId:",ids[shot][0],"Cheating:", round(probability, 2)*100, "%         ","Tick:",ticks[shot][0],"File:",files[shot][0])
                    counter += 1
                    sussies.append(names[shot][0])
                    with open('cheaters.csv','a',newline='\n',encoding='UTF-8')as f:
                        thewriter = csv.writer(f)
                        thewriter.writerow([names[shot][0],ids[shot][0],round(probability, 2)*100,ticks[shot][0],files[shot][0]])


if __name__ == "__main__":

    # input demos folder
    dem_folder = './demos/'
    
    pipe = sp.Popen(f'go run parser.go {dem_folder}', shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    res = pipe.communicate()
    for line in res[0].decode(encoding='utf-8').split('\n'):
        print(line)
    model = Model("data/data.csv")  # 189
    model.predict(batch_size=3000)