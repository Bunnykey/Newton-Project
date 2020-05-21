import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv("https://bit.ly/3g3X0ir")

dataset = df.values
X = dataset[:, 2:50]
Y = dataset[:, 1]

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.25)

model = Sequential()
model.add(Dense(30, input_dim=48, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, Y_train, epochs=200, batch_size=10)

Y_pred = model.predict(X_valid).flatten()

for i in range(10):
    realAns = Y_valid[i]
    predAns = Y_pred[i]
    print('Real : {}, Pred : {}'.format(realAns,predAns))