import pandas as pd
import torch
import torch.nn as nn
import subprocess as sp
import sys
import math

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
        X = data[:, 1:]
        features = [x for x in range(24)]
        num_features = len(features)
        idxs = []
        for i in range(X.shape[1]):
            if (i >= num_features * num_samples):
                continue
            if (i % num_features) in features:
                idxs.append(i)
        X = X.reshape(-1, 256, num_features)
        return X

    def predict(self,batch_size):

        X = self.extract_features()
        #X = X[0,:,:]
        # Shape expected is (n,256,24). Example has 20 shots so the shape is (20,256,24)
        print(f"Shape: {X.shape}")

        # Deep learning model. Remove map_location if GPU-enabled PyTorch. Enabling GPU speeds up predictions, but may
        # not be needed if predicting small amounts of games
        model = torch.load("AntiCheat2.pt",map_location=torch.device('cpu'))

        # rounded up n shots / batch size
        total_batches = math.ceil(X.shape[0] / batch_size)
        print(total_batches)
        for inx in range(total_batches):
            x = X[inx * batch_size:inx * batch_size + batch_size, :, :]
            x = torch.tensor(x).to(device).float()
            prd = model.forward(x)
            probs = torch.softmax(prd, 1)
            # Each shot in batch
            for shot in range(x.shape[0]):
                probability = probs[shot][1].item()

                # The probabilities are way too confident. For example use 95 % as threshold for a cheating shot.
                # You can come up with any rule you want, for example if average is over X% or if top 5 predictions are over
                # X% or even create a ML model on top of these

                if probability > 0.2:
                    print("Shot number:", shot, "Cheating:", round(probability, 2)*100, "%")


if __name__ == "__main__":
    dem_folder = './a/'
    pipe = sp.Popen(f'go run parser.go {dem_folder}', shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    res = pipe.communicate()
    for line in res[0].decode(encoding='utf-8').split('\n'):
        print(line)
    model = Model("./parsed_games/data.csv")

    # Make sure you can fit the data on RAM or VRAM if using GPU
    # each shot is around 50 000 bytes
    model.predict(batch_size=100)