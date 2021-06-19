from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def datacreator():
    y_train = []
    bigx = []
    X_train = []
    for folder in ["singlekills","cleankills"]:
        for x in os.listdir(f"D:/Users/emill/csgocheaters/{folder}"):


            df = pd.read_csv(f"D:/Users/emill/csgocheaters/{folder}/{x}",index_col=0)
            if len(df) > 20:
                df = df.select_dtypes(['number'])
                df = df.drop("sus",axis=1)   # junk
                chunk = []
                for i in range(20):
                    a = np.array(df.iloc[i])
                    X_train.append(a)
                bigx.append(X_train)
                if folder == 'singlekills':
                    y_train.append(1)
                elif folder == 'cleankills':
                    y_train.append(0)


    X_train = np.array(bigx)

    return X_train,y_train


X, y = datacreator()
#print(y)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("X",X_train.shape)
print("len y:",len(y_train))
y_train = np.asarray(y_train).astype('int').reshape((-1,1))
print(y_train)
print(y_train.shape)
# samples, time steps, and features.


model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 15)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(1,activation='sigmoid'))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 1)