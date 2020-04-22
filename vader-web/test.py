import os
from configparser import ConfigParser

import joblib
import numpy as np
import scipy.io as sio

from common_utils import load_meta_config
from jedi_dataset import JediDataset
from jedi_recommender import JEDI_harmonic, JEDI_blackbox

# load configuration
config = ConfigParser()
config.read('app.cfg')
dataset_config = config['TXT']

# # load metadata
# df_meta = load_meta_config(config['DEFAULT']['META_PATH'], 'dog')
#
# ADJ_PATH = dataset_config['ADJ_PATH']
# # load the data set.
# dataset = JediDataset('cat', features_path=dataset_config['DATA_PATH'])
#
# model = joblib.load(dataset_config['MODEL_PATH'])
# W = model.coefficients()
#
# train_X = np.concatenate((np.ones((dataset.train_X.shape[0], 1)), dataset.train_X), axis=1)
# test_X = np.concatenate((np.ones((dataset.test_X.shape[0], 1)), dataset.test_X), axis=1)
# X = np.vstack([train_X, test_X])
# Y = np.hstack([dataset.train_Y, model.predict(test_X)])

import pickle

# Load the Data.
tr_X = pickle.load(open(os.path.join(dataset_config['DATA_PATH'], 'features_train.pickle'), 'rb'))
te_X = pickle.load(open(os.path.join(dataset_config['DATA_PATH'], 'features_test.pickle'), 'rb'))
tr_y = pickle.load(open(os.path.join(dataset_config['DATA_PATH'], 'labels_train_flipped0.3.pickle'), 'rb'))
te_y = pickle.load(open(os.path.join(dataset_config['DATA_PATH'], 'labels_test_prediction.pickle'), 'rb'))
te_y = np.argmax(te_y, axis=1)
te_y = te_y*2.-1.

X = np.vstack([tr_X.todense(), te_X.todense()])
Y = np.hstack([tr_y, te_y])

ADJ_PATH = dataset_config['ADJ_PATH']



if os.path.exists(ADJ_PATH):
  A = joblib.load(ADJ_PATH)
else:
  import matlab
  import matlab.engine

  eng = matlab.engine.start_matlab()

  XMAT = matlab.double(X.tolist())

  A = eng.generateAffinityMatrix(XMAT)
  A = np.asarray(A)
  joblib.dump(A, ADJ_PATH, compress=3)

# # influence scores.
# inf_scores = joblib.load(config['CATS']['INF_RESULTS_PATH'])
# I = np.mean(np.abs(inf_scores), axis=1)
#
# alpha = 1.
#
# step = .2
# beta = 0.833
#
# indices = list(np.random.randint(0, 211, 10))
# YL = list(np.random.randint(0, 2, 10)*2-1)
#
# PL = np.random.uniform(0.01, 0.99, 10)
# PL = list(np.vstack([PL, 1. - PL]).T)
#
# print(YL)
#
# #
# prob  = JEDI_harmonic(Y, A, indices, YL)
# # print(prob)
#
#
# prob, fvalue = JEDI_blackbox(X, Y, W, step, A, beta, indices, PL, YL )
#
#
# P_JEDI = np.max(prob, axis=1)
# P_JEDI = P_JEDI / np.sum(P_JEDI)
# P_INF = I/I.sum()
#
#
# alpha = 1.
# P = P_INF + alpha*P_JEDI
#
#
# # load influence influence scores.
# inf_scores = joblib.load(dataset_config['INF_RESULTS_PATH'])
# I = np.mean(np.abs(inf_scores), axis=1)
# P_INF = I/I.sum()
#
#
# alpha = 1.
# P = P_INF + alpha*P_JEDI



# Filter and return the new examples.






# print(prob.shape)
# print(fvalue.shape)
#
# print(prob)
#
#
# print(fvalue)

# for i in range(5):


# _ysl = np.abs(selectProb)
# _ysl = _ysl / np.max(_ysl)
# _ysl[_ysl<1] = 0
#
# _ysl_prob = np.abs(selectProb/np.sum(selectProb))
#
# ysl_prob += [_ysl_prob]
# ysl += [_ysl]
#
# # ysl.append(_ysl)
# # ysl_prob.append()
# order.append(selectIdx)
#
#
# print(selectIdx)
