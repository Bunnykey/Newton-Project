from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import functools
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from IPython.display import clear_output
from six.moves import urllib

#generating trainset and testset

dataURL = "https://bit.ly/3g3X0ir"

# df = pd.read_csv(dataURL)

# df['split'] = np.random.randn(df.shape[0], 1)
# trainSplit = np.random.rand(len(df)) <= 0.7

# blueWins = df.pop('blueWins')
# dataset = tf.data.Dataset.from_tensor_slices((df.values, blueWins.values))
# trainSet = df[trainSplit]
# testSet = df[~trainSplit]

# column_names = ['gameId','gameDuraton','blueWins','blueFirstBlood',
#                 'blueFirstTower','blueFirstBaron','blueFirstDragon',
#                 'blueFirstInhibitor','blueDragonKills','blueBaronKills',
#                 'blueTowerKills','blueInhibitorKills','blueWardPlaced',
#                 'blueWardkills','blueKills','blueDeath','blueAssist',
#                 'blueChampionDamageDealt','blueTotalGold','blueTotalMinionKills',
#                 'blueTotalLevel','blueAvgLevel','blueJungleMinionKills',
#                 'blueKillingSpree','blueTotalHeal','blueObjectDamageDealt',
#                 'redWins','redFirstBlood','redFirstTower','redFirstBaron','redFirstDragon',
#                 'redFirstInhibitor','redDragonKills','redBaronKills','redTowerKills',
#                 'redInhibitorKills','redWardPlaced','redWardkills','redKills','redDeath',
#                 'redAssist','redChampionDamageDealt','redTotalGold','redTotalMinionKills',
#                 'redTotalLevel','redAvgLevel','redJungleMinionKills','redKillingSpree',
#                 'redTotalHeal','redObjectDamageDealt']

raw_dataset = pd.read_csv(dataURL)

dataset = raw_dataset.copy()

trainSet = dataset.sample(frac=0.7,random_state=0)
testSet = dataset.drop(trainSet.index)
trainLabel = trainSet.pop('blueWins')
testLabel = testSet.pop('blueWins')

trainStats = trainSet.describe()
trainStats = trainStats.transpose()

print(trainStats)

def norm(x):
    return (x - trainStats['mean']) / trainStats['std']

normedTrainData = norm(trainSet)
normedTestData = norm(testSet)

trainSet.gameDuraton.hist(bins=20)
# plt.show() # most games end in 2500sec

def build_model():
  model = keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[len(trainSet.keys())]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
# print(model.summary())

example_batch = normedTrainData[:10]
#example_result = model.predict(example_batch)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: 
        print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normedTrainData, trainLabel,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)