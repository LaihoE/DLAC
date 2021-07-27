import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,precision_score
import pickle


df1 = pd.read_csv(r"D:\csgo\new_faceit_s/dirty_scaled.csv",header=None)
print(len(df1))
df2 = pd.read_csv(r"D:\csgo\new_faceit_s/clean_scaled.csv",header=None)


df = df1.append(df2,ignore_index=True)
y = [1 if x<29101 else 0 for x in range(len(df))]

print(y)
print(df)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=42)

cb_modelh = CatBoostClassifier(iterations=400,
                              depth=6,
                              task_type="CPU",
                              eval_metric='Accuracy',
                              random_seed=42,
                              od_type='Iter',
                              metric_period=10,
                              od_wait=10
                              )
cb_modelh.fit(X_train, y_train, verbose=10)


yhat = cb_modelh.predict(X_test)
# evaluate predictions
acc = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % acc)

acc = precision_score(y_test, yhat)
print('Accuracy: %.3f' % acc)

with open("cbmodel2.model", "w+b") as f:
    pickle.dump(cb_modelh, f)