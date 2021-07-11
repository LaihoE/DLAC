import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from math import pi

def datacreator():
    y_train = []
    bigx = []
    cnt=0
    #
    for folder in ["singlekills","b","c","d","e","f","1","2"]:
        for x in os.listdir(f"D:/Users/emill/csgocheaters/singlekills/{folder}"):

            df = pd.read_csv(f"D:/Users/emill/csgocheaters/singlekills/{folder}/{x}",index_col=0)

            #df = pd.read_csv(
                #f"C:/Users/emill/PycharmProjects/open_anti_cheat/cleandrity/singlekills/{file_info[0]}", index_col=0)
            df = df.select_dtypes(['number'])
            df = df.drop("sus", axis=1)  # junk

            limitPer = len(df) * .80
            df = df.dropna(thresh=limitPer, axis=1)
            cnt+=1
            print(cnt/5350*100)
            #
            # TRIGONOMERTICS
            #
            try:
                df = df[["X_x", "Y_x", "Z_x", "ViewX_x", "ViewY_x","X_y", "Y_y", "Z_y", "ViewX_y", "ViewY_y"]]

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


                df[["X_Off_By_Degrees_x","Y_Off_By_Degrees_x","Y_Off_By_Degrees_y","X_Off_By_Degrees_y","X_x", "Y_x", "Z_x", "ViewX_x", "ViewY_x","X_y", "Y_y", "Z_y", "ViewX_y", "ViewY_y"]] = scaler.fit_transform(df[["X_Off_By_Degrees_x","Y_Off_By_Degrees_x","Y_Off_By_Degrees_y","X_Off_By_Degrees_y","X_x", "Y_x", "Z_x", "ViewX_x", "ViewY_x","X_y", "Y_y", "Z_y", "ViewX_y", "ViewY_y"]])
                #df[["ViewX", "ViewY"]] = scaler.fit_transform(df[["ViewX", "ViewY"]])

                if df["Y_Off_By_Degrees_x"].isna().sum() == 0:
                    df = df.select_dtypes(['number'])
                    #df = df.drop("sus",axis=1)   # junk

                    X = np.array(df.iloc[len(df) - 200:])
                    bigx.append(X)
                    if folder == 'singlekills':
                        y_train.append(1)
                    elif folder == 'a':
                        y_train.append(1)
                    elif folder == 'b':
                        y_train.append(1)
                    elif folder == 'c':
                        y_train.append(1)
                    elif folder == 'd':
                        y_train.append(1)
                    elif folder == 'e':
                        y_train.append(1)
                    elif folder == 'f':
                        y_train.append(1)


                    elif folder == '1':
                        y_train.append(0)
                    elif folder == '2':
                        y_train.append(0)
                    elif folder == '3':
                        y_train.append(0)
                    elif folder == '4':
                        y_train.append(0)
            except Exception as e:
                print(e)



    X_train = np.array(bigx)
    return X_train,y_train

def datacreator_testing():
    y_train = []
    bigx = []

    for folder in ["singlekills","cleankills"]:
        for x in os.listdir(f"D:/Users/emill/csgocheaters/testing/{folder}"):

            df = pd.read_csv(f"D:/Users/emill/csgocheaters/testing/{folder}/{x}",index_col=0)

            #df = pd.read_csv(
                #f"C:/Users/emill/PycharmProjects/open_anti_cheat/cleandrity/singlekills/{file_info[0]}", index_col=0)
            df = df.select_dtypes(['number'])
            df = df.drop("sus", axis=1)  # junk

            df = df[["X_x", "Y_x", "Z_x", "ViewX_x", "ViewY_x", "X_y", "Y_y", "Z_y", "ViewX_y", "ViewY_y"]]

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

            df[["X_Off_By_Degrees_x", "Y_Off_By_Degrees_x", "Y_Off_By_Degrees_y", "X_Off_By_Degrees_y", "X_x", "Y_x",
                "Z_x", "ViewX_x", "ViewY_x", "X_y", "Y_y", "Z_y", "ViewX_y", "ViewY_y"]] = scaler.fit_transform(df[[
                "X_Off_By_Degrees_x", "Y_Off_By_Degrees_x", "Y_Off_By_Degrees_y", "X_Off_By_Degrees_y", "X_x", "Y_x",
                "Z_x", "ViewX_x", "ViewY_x", "X_y", "Y_y", "Z_y", "ViewX_y", "ViewY_y"]])

            if df["Y_Off_By_Degrees_x"].isna().sum() == 0:
                df = df.select_dtypes(['number'])
                #df = df.drop("sus",axis=1)   # junk

                X = np.array(df.iloc[len(df) - 200:])
                bigx.append(X)
                if folder == 'singlekills':
                    y_train.append(1)
                elif folder == 'cleankills':
                    y_train.append(0)

    X_train = np.array(bigx)
    return X_train,y_train



X_train,y_train = datacreator()
print(X_train.shape)
print(y_train)


np.save("X_train",X_train)
np.save("y_train",y_train)
print("-----TESTING------")
X_train,y_train = datacreator_testing()
print(X_train.shape)
print(y_train)

np.save("X_train_testing",X_train)
np.save("y_train_testing",y_train)