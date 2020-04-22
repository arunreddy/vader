import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import logging
import datetime
import numpy as np 
import tensorflow as tf 
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.enable_eager_execution()

from jedi_dataset import JediDataset
from logistic_regression import LogisticRegression
from lr_unbiased_estimator import LR_UnbiasedEstimator
from influence import Influence

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main(features_path, dataset_name, v_type, loss_type=1, debug=False, plot_results=False):

    # Load the jedi dataset.
    dataset = JediDataset(features_path=features_path, name=dataset_name)
    NUM_FEATURES = dataset.train_X.shape[1]
    
    images_path = '/home/arun/research/projects/crowdsourcing/kdd-2019/data/cats_dogs/all'

    # Build the classifier.
    # TODO: Modify the code to use the LR unbiased estimator.

    model = LR_UnbiasedEstimator(setting=loss_type)
    model.fit(dataset.train_X, dataset.train_Y)
    W = model.coefficients()

    dataset.test_Y = model.predict(dataset.test_X)

    assert W.shape[0] == NUM_FEATURES

    tf.reset_default_graph()

    v_W = tf.constant(W, name='w', dtype=tf.float32)

    if loss_type == 0:
        influence = Influence(W=v_W, loss_type='logistic_loss')
    else: 
        influence = Influence(W=v_W, loss_type='surrogate_loss')

    num_train = dataset.train_X.shape[0]
    num_test = dataset.test_X.shape[0]


    # compute class weights.
    class_weights = {}
    unique_classes = np.unique(dataset.train_Y, return_counts=True)
    for idx in range(unique_classes[0].shape[0]):
        c = unique_classes[0][idx]
        v = unique_classes[1][idx]/num_train
        class_weights[c] = v


    # Compute the influence on perturbation loss 
    inf_pert_loss = np.zeros(shape=[num_test, num_train])

    if v_type == 'train':
        tgt_indices = np.arange(num_train)
        inf_pert_loss = np.zeros(shape=[num_train, num_train])
    elif v_type == 'test':
        tgt_indices = np.arange(num_test)


    for test_idx in tgt_indices:
        start = datetime.datetime.now()
        inv_hvp = influence.inv_hvp_lissa(dataset, v_idx=test_idx, v_type=v_type)
        end = datetime.datetime.now()
        exec_time = (end-start).total_seconds()
        print('===== Executed in  {:0.2f} seconds ====='.format(exec_time))

        influence_on_training_points = []
        # compute the influence of each training points.
        for train_idx in range(num_train):
            tr_X, tr_Y = dataset.fetch_train_instance(train_idx)

            v_X = tf.Variable(tr_X, 'X', dtype=tf.float32)
            v_Y = tf.Variable(tr_Y, 'Y', dtype=tf.float32)

            dl_dw = influence.dl_dw(v_X, v_Y, v_W)
            dl_dydw = influence.dl_dydw(v_X, v_Y, v_W).numpy()
            a = -1* np.tensordot(dl_dydw, inv_hvp, axes=1).flatten()[0]


            _influence = a*class_weights[int(tr_Y)]

            influence_on_training_points.append(_influence)

            inf_pert_loss[test_idx, train_idx] = _influence

        # np.savez_compressed('./cache/inf_of_test_{}_{}_{}_{}.npz'.format(v_type, dataset_name, loss_type, test_idx), influence = _influence)

    np.savez_compressed('./cache/{}/inf_pert_loss_{}_{}_{}.npz'.format(dataset_name, v_type, dataset_name, loss_type),
                        inf_pert_loss=inf_pert_loss)

    print('**************************** COMPLETE *****************************************')

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Compute Influence.')
    parser.add_argument('-l','--logistic_loss', help='Logistic Loss Loss', action='store_true')
    parser.add_argument('-u','--unbiased_loss', help='Unbiased Loss', action='store_true')

    parser.add_argument('-t','--train', help='train_examples', action='store_true')

    args = vars(parser.parse_args())


    features_path = '/home/arun/research/projects/crowdsourcing/kdd-2019/data/animal_breed_sdm/data_dog_flipped0.2'
    dataset = 'dog'
    v_type = 'test'
    if args['train']:
        v_type = 'train'

    if args['logistic_loss']:
        main(features_path, dataset, v_type=v_type, loss_type=0)
    else:
        main(features_path, dataset, v_type=v_type, loss_type=1)