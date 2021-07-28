import pandas as pd
import torch
import torch.nn as nn
import subprocess as sp
import math
import numpy as np
import joblib


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
        my_scaler = joblib.load(r'C:\Users\emill/scaler.gz')
        self.df = pd.read_csv(df_path,header=None)
        print(self.df)
        self.df = self.df.values  # returns a numpy array
        print(self.df)
        self.df = my_scaler.transform(self.df)
        self.df = pd.DataFrame(self.df)
        print(self.df)
        self.X = None


    def extract_features(self):
        num_samples = len(self.df)
        data = self.df.to_numpy()
        X = data[:, 1:]
        y = data[:, 0]

        features = [x for x in range(24)]
        num_features = len(features)
        idxs = []
        for i in range(X.shape[1]):
            if (i >= num_features * num_samples):
                continue
            if (i % num_features) in features:
                idxs.append(i)

        X = X.reshape(-1, 256, num_features)
        return X,y

    def predict(self,batch_size):

        X,y = self.extract_features()
        #name = X[:,:,0]
        #id = X[:,:,2]
        #tick = X[:,:,1]

        # Drop the id and tick since it's not fed into the model
        #X = X[:,:,3:]
        # Shape expected is (n,256,24). Example has 20 shots so the shape is (20,256,24)
        print(f"Shape: {X.shape}")
        print(X[0])
        X = X.astype(np.float32)
        # Deep learning model. Remove map_location if GPU-enabled PyTorch. Enabling GPU speeds up predictions, but may
        # not be needed if predicting small amounts of games
        model = torch.load("C:/Users/emill/PycharmProjects/CLIP/canmodel/grus/AntiCheat14.pt",map_location=torch.device('cpu'))

        # rounded up n shots / batch size
        total_batches = math.ceil(X.shape[0] / batch_size)

        for inx in range(total_batches):
            x = X[inx * batch_size:inx * batch_size + batch_size, :, :]
            #ids = id[inx * batch_size:inx * batch_size + batch_size,0]
            #ticks = tick[inx * batch_size:inx * batch_size + batch_size,:]
            #names = name[inx * batch_size:inx * batch_size + batch_size]
            #print(names)
            x = torch.tensor(x).to(device).float()
            prd = model.forward(x)
            probs = torch.softmax(prd, 1)
            # Each shot in batch
            for shot in range(x.shape[0]):
                probability = probs[shot][1].item()
                # The probabilities are way too confident. For example use 95 % as threshold for a cheating shot.
                # You can come up with any rule you want, for example if average is over X% or if top 5 predictions are over
                # X% or even create a ML model on top of these

                if probability > 0.95:                                        # sampled 64-tick so need to *2 if 128 server
                    #print("Name",names[shot][0],"SteamID:", str(ids[shot]),"Tick",(ticks[shot,0]* 2), "Cheating:", round(probability, 2)*100, "%")
                    print("Cheating:", round(probability, 2)*100, "%")


if __name__ == "__main__":
    dem_folder = './demos2/'
    pipe = sp.Popen(f'go run parser.go {dem_folder}', shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    res = pipe.communicate()
    for line in res[0].decode(encoding='utf-8').split('\n'):
        print(line)
    model = Model("data/data.csv")
    model.predict(batch_size=100)