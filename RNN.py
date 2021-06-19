#from keras.models import Sequential
#from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import pandas as pd
import os
import numpy as np

X_train = []
y_train = []
bigx = []

for x in os.listdir(r"D:\Users\emill\csgocheaters\singlekills"):
    X_train = []
    y_train = []
    df = pd.read_csv(f"D:/Users/emill/csgocheaters/singlekills/{x}",index_col=0)
    df = df.select_dtypes(['number'])
    df = df.drop("sus",axis=1)   # junk
    chunk = []
    for i in range(20):
        a = np.array(df.iloc[i])
        X_train.append(a)
    bigx.append(X_train)

print(np.array(bigx).shape)