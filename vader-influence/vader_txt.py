import os
import pickle
import numpy as np
from lr_unbiased_estimator import LR_UnbiasedEstimator
import joblib
from math import exp
import tensorflow as tf
import argparse
import tempfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.enable_eager_execution()
import datetime
from jedi_txt_dataset import JediTextDataset

from influence import Influence

force_refresh = False

DATA_PATH = "/home/arun/Dropbox (ASU)/code_for_Arun/text_data/comp_vs_sci_flipped0.3/"
CACHE_DIR = "/home/arun/research/projects/crowdsourcing/kdd-2019/cache/txt"

dataset_name = 'txt'
loss_type = 'unbiased'


def __sigmoid( z):
    if z >= 0:
        return 1 / (1 + exp(-z))
    else:
        return exp(z) / (1 + exp(z))

tr_X = np.asarray(pickle.load(open(os.path.join(DATA_PATH, 'features_train.pickle'), 'rb')).todense())
te_X = np.asarray(pickle.load(open(os.path.join(DATA_PATH, 'features_test.pickle'), 'rb')).todense())

tr_y = pickle.load(open(os.path.join(DATA_PATH, 'labels_train_flipped0.3.pickle'), 'rb'))
te_y = pickle.load(open(os.path.join(DATA_PATH, 'labels_test_prediction.pickle'), 'rb'))
te_y = np.argmax(te_y, axis=1)
te_y_orig = te_y*2.-1.



tr_X = np.concatenate((np.ones((tr_X.shape[0], 1)), tr_X), axis=1)
te_X = np.concatenate((np.ones((te_X.shape[0], 1)), te_X), axis=1)



nD = tr_X.shape[1]

print('Generating the model')

model_path = os.path.join(CACHE_DIR, 'model_{}_{}.dat'.format(dataset_name, loss_type))

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = LR_UnbiasedEstimator(loss_type=loss_type, fit_intercept=False)
    model.fit(tr_X, tr_y)
    joblib.dump(model, model_path, compress=3)

# Load the model coefficients.
W = model.coefficients()
assert W.shape[0] == nD


# populate the test labels using the logistic regression model.
# The value is continuous in the interval [-1,1]
pY = model.predict_prob(te_X)
te_y = (pY * 2.) - 1.

tf.reset_default_graph()
vW = tf.constant(W, name='w', dtype=tf.float32)
influence = Influence(W=vW, loss_type=loss_type)

num_train = tr_X.shape[0]

# compute class weights.
class_weights = {}
unique_classes = np.unique(tr_y, return_counts=True)
for idx in range(unique_classes[0].shape[0]):
    c = unique_classes[0][idx]
    v = unique_classes[1][idx] / num_train
    class_weights[c] = v

# Merge train and test data.
X = np.vstack([tr_X, te_X])
Y = np.hstack([tr_y, te_y])

# Compute the marginal distance.
marginal_distance = np.abs(np.dot(X, W)) / np.linalg.norm(W, ord=2)
marginal_distance_path = os.path.join(CACHE_DIR, 'marginal_distance_{}_{}.dat'.format(dataset_name, loss_type))
joblib.dump(marginal_distance, marginal_distance_path, compress=3)

dataset = JediTextDataset('jedi_txt', tr_X, tr_y, te_X, te_y)


list_of_idx = np.arange(X.shape[0])

inf_pert_loss = np.zeros(shape=[X.shape[0], num_train])
is_flipped = np.zeros(shape=X.shape[0])

for idx in list_of_idx:

    # compute the gradient w.r.t the given example. Please note that the given example could be test or train point in our framework.
    x = X[idx, :]
    y = Y[idx]

    vX = tf.Variable(x, 'test_x', dtype=tf.float32)
    vY = tf.Variable(y, 'test_y', dtype=tf.float32)

    test_dl_dw = influence.dl_dw(vX, vY, vW)

    # file_name = all_names[idx].split('.')[0]
    # _df = df[df.common_name == file_name].values[0]
    # if _df[5] == 'flipped':
    #     is_flipped[idx] = 1

    # compute the hvp (hessian vector product) using the gradient for the given example.
    cache_file = os.path.join(os.path.join(CACHE_DIR, 'inv_hvp', 'inv_hvp_{}_{}.npz'.format(loss_type, idx)))
    if not force_refresh and os.path.exists(cache_file):
        print('Loading HVP file for idx {} from cache at {}'.format(idx, cache_file))
        inv_hvp = np.load(cache_file)['inv_hvp']
    else:
        start = datetime.datetime.now()
        inv_hvp = influence.inv_hvp_lissa_fast(dataset, test_dl_dw)
        end = datetime.datetime.now()
        exec_time = (end - start).total_seconds()
        print('Saving HVP file for idx {} to cache at {}'.format(idx, cache_file))
        np.savez_compressed(cache_file, inv_hvp=inv_hvp)

    # compute the influence of each training example on the given example
    influence_on_training_points = []
    # compute the influence of each training points.
    for train_idx in range(num_train):
        trX, trY = dataset.fetch_train_instance(train_idx)

        vX = tf.Variable(trX, 'X', dtype=tf.float32)
        vY = tf.Variable(trY, 'Y', dtype=tf.float32)

        dl_dydw = influence.dl_dydw(vX, vY, vW).numpy()
        a = -1 * np.tensordot(dl_dydw, inv_hvp, axes=1).flatten()[0]

        _influence = a * class_weights[int(trY)]

        influence_on_training_points.append(_influence)

        inf_pert_loss[idx, train_idx] = _influence

    # save the results to the disk.
    results_file = os.path.join(CACHE_DIR, 'inf_scores',
                                'influence_scores_{}_{}_{}_{}.dat'.format(idx, dataset_name, loss_type,
                                                                          is_flipped[idx]))
    joblib.dump(inf_pert_loss[idx, :], results_file, compress=3)
    print('Saving the influence scores on all the trainign poitns for idx {} to {}'.format(idx, results_file))

# save the results to the disk.
results_file = os.path.join(CACHE_DIR, 'influence_scores_{}_{}.dat'.format(dataset_name, loss_type))
joblib.dump(inf_pert_loss, results_file, compress=3)
print('Saving the perturbation loss results to the disk at {}'.format(results_file))

is_flipped_file = os.path.join(CACHE_DIR, 'example_flipped_{}_{}.dat'.format(dataset_name, loss_type))
joblib.dump(is_flipped, is_flipped_file, compress=3)
print('Saving the flipped data to the disk at {}'.format(is_flipped_file))