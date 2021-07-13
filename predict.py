import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from csgo.parser import DemoParser
from no_path_parser import Killparser
import os
import numpy as np
import pandas as pd
from math import pi


scaler = MinMaxScaler()
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)


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

    def predict(self,x):
        model.eval()
        with torch.no_grad():
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




def dem_to_json(path):
    cwd = os.getcwd()
    demo_parser = DemoParser(demofile=path, log=False, parse_rate=1,outpath=f'{cwd}/json_files')
    demo_parser.parse()

def split_correct_kills(json_name, steamid):
    cwd = os.getcwd()
    datamover = Killparser()
    datamover.read_json(f"{cwd}/json_files/{json_name}")
    datamover.get_pos_player(steamid,f"{cwd}/killdata/games/")
    datamover.get_kills_csv(f"{cwd}/killdata/kills/")
    json_name_without_end = json_name.replace(".json","")
    mainfile = f'{cwd}/killdata/games/{json_name_without_end}.csv'
    killfile = f'{cwd}/killdata/kills/{json_name_without_end}.csv'
    out_folder = f'{cwd}/killdata/suspects_kills/'
    datamover.split_df_by_kill(mainfile, killfile, steamid, out_folder, write_to_csv=True)


if __name__ == "__main__":
    # Get current working directory
    cwd = os.getcwd()
    print(cwd)
    path_to_dem_file = 'F:/csgo/b/'
    dem_file_name = 'aaf803c6-71d3-47e8-bb28-9b61ac30dfd3.dem'
    steam_id = 76561199158549813

    # Call function to parse .dem file into json. This part takes by far the longest

    try:
        dem_to_json(path_to_dem_file+dem_file_name)
    except Exception as e:
        print("Error reading demo file")
        print(e)
    print("NOT HERER")
    json_file_name = dem_file_name.replace(".dem",".json")
    split_correct_kills(json_file_name,steam_id)





    model = torch.load("AntiCheat.pt", map_location=torch.device('cpu'))
    for file in os.listdir(f'{cwd}/killdata/suspects_kills/'):
        df = pd.read_csv(f"{cwd}/killdata/suspects_kills/{file}")
        limitPer = len(df) * .80
        df = df.dropna(thresh=limitPer, axis=1)

        #
        # TRIGONOMERTICS
        #
        try:
            df = df[["IsAirborne_x", "IsFlashed_x", "IsScoped_x", "IsWalking_x", "Money_x",
                     "Money_y", "X_x",
                     "Y_x", "Z_x", "ViewX_x", "ViewY_x", "X_y", "Y_y", "Z_y", "ViewX_y", "ViewY_y"]]

            # Trigonometrics for killer
            df['ViewX_x'] = (df['ViewX_x'] + 180) % 360 - 180
            df['ViewY_x'] = (df['ViewY_x'] + 180) % 360 - 180

            x1 = df["X_x"]
            y1 = df["Y_x"]
            z1 = df["Z_x"]

            x2 = df["X_y"]
            y2 = df["Y_y"]
            z2 = df["Z_y"]

            xdif = np.array(x2 - x1)
            ydif = np.array(y2 - y1)
            zdif = np.array(z2 - z1)

            rightLeft = np.arctan2(ydif, xdif) * (180 / pi)
            hypot3D = np.sqrt(zdif ** 2 + xdif ** 2 + ydif ** 2)
            upDown = (-np.arcsin(zdif / hypot3D)) * (180 / pi)

            df["X_Off_By_Degrees_x"] = rightLeft - df["ViewX_x"]
            df["Y_Off_By_Degrees_x"] = upDown - df["ViewY_x"]

            # Trigonometrics for victim
            df['ViewX_y'] = (df['ViewX_y'] + 180) % 360 - 180
            df['ViewY_y'] = (df['ViewY_y'] + 180) % 360 - 180

            x1 = df["X_y"]
            y1 = df["Y_y"]
            z1 = df["Z_y"]

            x2 = df["X_x"]
            y2 = df["Y_x"]
            z2 = df["Z_x"]

            xdif = np.array(x2 - x1)
            ydif = np.array(y2 - y1)
            zdif = np.array(z2 - z1)

            rightLeft = np.arctan2(ydif, xdif) * (180 / pi)
            hypot3D = np.sqrt(zdif ** 2 + xdif ** 2 + ydif ** 2)
            upDown = (-np.arcsin(zdif / hypot3D)) * (180 / pi)

            df["X_Off_By_Degrees_y"] = rightLeft - df["ViewX_x"]
            df["Y_Off_By_Degrees_y"] = upDown - df["ViewY_x"]

            df[["X_Off_By_Degrees_x", "Y_Off_By_Degrees_x", "X_Off_By_Degrees_y", "Y_Off_By_Degrees_y", "IsAirborne_x",
                "IsFlashed_x", "IsScoped_x", "IsWalking_x", "Money_x",
                "Money_y", "X_x",
                "Y_x", "Z_x", "ViewX_x", "ViewY_x", "X_y", "Y_y", "Z_y", "ViewX_y", "ViewY_y"]] = scaler.fit_transform(df[[
                "X_Off_By_Degrees_x", "Y_Off_By_Degrees_x", "IsAirborne_x", "IsFlashed_x", "IsScoped_x", "IsWalking_x",
                "Money_x",
                "Money_y", "X_x", "X_Off_By_Degrees_y", "Y_Off_By_Degrees_y",
                "Y_x", "Z_x", "ViewX_x", "ViewY_x", "X_y", "Y_y", "Z_y", "ViewX_y", "ViewY_y"]])


            X = np.array(df.iloc[len(df) - 300:])
            X = X.reshape((1,300,20))
            X = torch.tensor(X)
            X = X.to(device)
            X = X.float()

            prd = model.forward(X)
            probs = torch.softmax(prd, 1)

            print("Probability of cheating:", round(probs[0][0].item(), 2), "            Logits:", prd)


        except Exception as e:
            print(e)