import pandas as pd
import subprocess as sp
import pickle
import csv
import catboost


class Model:
    def __init__(self,df_path,data_path,shots_file_map):
        self.df = pd.read_csv(df_path,header=None)
        self.shots_file_map = shots_file_map
        self.data = pd.read_csv(data_path,header=None)
        self.X = None

    def extract_features(self):
        num_samples = len(self.df)
        data = self.df.to_numpy()
        X = data[:, 1:]
        y = data[:, 0]

        features = [x for x in range(27)]
        num_features = len(features)
        idxs = []
        for i in range(X.shape[1]):
            if (i >= num_features * num_samples):
                continue
            if (i % num_features) in features:
                idxs.append(i)

        X = X.reshape(-1, 256, num_features)
        return X,y

    def predict(self):

        X,y = self.extract_features()
        name = X[:,:,0]
        id = X[:,:,2]
        tick = X[:,:,1]

        with open("cb_model.model", "rb") as f:
            cb_model = pickle.load(f)

        prd = cb_model.predict_proba(self.data)
        print("Total shots:",len(self.data))
        print("-----Predictions--------")
        # Create Headers
        with open('suspects.csv', 'w', newline='\n', encoding='UTF-8')as f:
            thewriter = csv.writer(f)
            thewriter.writerow(["Probability", "Name", "SteamId","TICK", "Demo_file_name"])

        sus_shots = []
        for x in range(len(prd)):
            single_kill = prd[x][1]
            if single_kill > 0.95:     # THRESHOLD, slightly too confident to be interpreted like a probability (goes from 0-1)
                sus_shots.append(x)

        for i in sus_shots:
            probability = prd[i][-1]
            kill_tick = tick[i][-1]
            kill_name = name[i][-1]
            kill_id = id[i][-1]
            file_name = ""
            for f in self.shots_file_map:
                if f[1]>i:
                    file_name = f[0]
                    break

            print("Probability:",round(probability,2),"Name:",kill_name,"SteamId:",kill_id,"Tick:",kill_tick*2)

            with open('suspects.csv','a',newline='\n',encoding='UTF-8')as f:
                thewriter = csv.writer(f)
                thewriter.writerow([round(probability,3),kill_name,kill_id,kill_tick*2,file_name])


if __name__ == "__main__":

    demo_folder = './demo_folder/'

    pipe = sp.Popen(f'go run main.go {demo_folder}', shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    res = pipe.communicate()
    filenames = []
    shots = []

    for line in res[0].decode(encoding='utf-8').split('\n'):
        if ".dem" in line:
            filenames.append(line)
        else:
            try:
                shots.append(line.split(":")[1])
            except Exception as e:
                print(e)
    shots_file_map = []
    totalshots = 0
    for i in range(len(filenames)):
        shots_this_file = shots[i]
        totalshots += int(shots_this_file)
        shots_file_map.append((filenames[i],totalshots))

    # Could be combined with above
    pipe = sp.Popen(f'go run tickparser.go {demo_folder}', shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    res = pipe.communicate()
    for line in res[0].decode(encoding='utf-8').split('\n'):
        print(line)

    model = Model("data/dataticks.csv","data/data.csv",shots_file_map)
    model.predict()