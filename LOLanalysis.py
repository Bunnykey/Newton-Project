from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import functools
import pathlib
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

dataURL = "https://bit.ly/3g3X0ir"

raw_dataset = pd.read_csv(dataURL)
dataset = raw_dataset.copy()
dataset.dropna()

trainSet = dataset.sample(frac=0.7,random_state=0)
testSet = dataset.drop(trainSet.index)

sns.pairplot(trainSet[["blueWins", "gameDuraton","blueFirstBlood","blueFirstTower"]], diag_kind="kde")
#plt.show()

trainStats = trainSet.describe()
trainStats.pop("gameDuraton")
trainStats = trainStats.transpose()

trainLabels = trainSet.pop('blueWins')
test_labels = testSet.pop('blueWins')

# exBatch = normedTrainData[:10]
# exResult = model.predict(exBatch)
# print(exResult)

target = trainSet.pop('gameDuraton')
tfTrainset = tf.data.Dataset.from_tensor_slices((trainSet.values, target.values))
newTrainSet = tfTrainset.shuffle(len(trainSet)).batch(1)

def get_compiled_model():
  model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(trainSet.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(optimizer=optimizer,
                loss='mse',
                metrics=['mae','mse'])
  return model

def norm(x):
  return (x - trainStats['mean']) / trainStats['std']

normedTestData = norm(testSet)

model = get_compiled_model()
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
history = model.fit(newTrainSet, epochs=20, callbacks=[early_stop])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

def plot_history(history):
  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)

loss, mae, mse = model.evaluate(normedTestData, test_labels, verbose=2)

test_predictions = model.predict(normedTestData).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

