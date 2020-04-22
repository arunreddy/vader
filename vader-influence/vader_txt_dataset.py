import sys
import os
import logging
import pandas as pd
import numpy as np
import pickle
from glob import glob
from sklearn.metrics import accuracy_score


class JediTextDataset(object):

  def __init__(self, name, train_X, train_Y, test_X, test_Y, noise_rate=0.3):
    self.logger = logging.getLogger(__name__)
    self.name = name

    self.label_map = {'domestic': 1, 'wild': -1}

    self.train_X = train_X
    self.train_Y = train_Y
    self.train_Y_ORIG = None

    self.test_X = test_X
    self.test_Y = test_Y
    self.test_Y_ORIG = None

    self.train_file_names = []
    self.test_file_names = []

    self.noise_rate = noise_rate

  def fetch_train_batch(self, batch_size):

    n_train = self.train_X.shape[0]
    assert batch_size <= self.train_X.shape[0]

    # Shuffle the data
    idx = np.arange(n_train)
    np.random.shuffle(idx)

    batch_idx = idx[:batch_size]

    _train_X = self.train_X[batch_idx, :].copy()
    _train_Y = self.train_Y[batch_idx].copy()

    return _train_X, _train_Y

  def fetch_test_instance(self, idx):
    test_X = self.test_X[idx, :]
    if len(test_X.shape) < 2:
      test_X = test_X.reshape(-1, 1).transpose()

    # return a matrix of the size 1xNUM_FEATURES
    assert test_X.shape[0] == 1
    return test_X, self.test_Y[idx]

  def fetch_train_instance(self, idx):
    train_X = self.train_X[idx, :]
    if len(train_X.shape) < 2:
      train_X = train_X.reshape(-1, 1).transpose()

    # return a matrix of the size 1xNUM_FEATURES
    assert train_X.shape[0] == 1
    return train_X, self.train_Y[idx]

