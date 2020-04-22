import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import logging
import datetime
import numpy as np 
import tensorflow as tf 
import argparse
import tempfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.enable_eager_execution()
import joblib

from jedi_dataset import JediDataset
from logistic_regression import LogisticRegression
from lr_unbiased_estimator import LR_UnbiasedEstimator
from influence import Influence

from tqdm import tqdm

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

from tqdm import  tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.io as sio
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

def main(features_path, dataset_name, loss_type, test_mode=False, force_refresh=False):


    # Check and create needed files.
    cache_dir = os.getenv('CACHE_DIR', None)
    if not cache_dir:
        cache_dir = '/tmp/influence-cache'
        os.mkdir(cache_dir)

    dataset_cache_dir = os.path.join(cache_dir, dataset_name)

    results_dir = os.getenv('RESULTS_DIR')
    if not results_dir:
        results_dir = '/tmp/influence-results'
        os.mkdir(results_dir)
    dataset_results_dir = os.path.join(results_dir, dataset_name)


    if not os.path.exists(dataset_cache_dir):
        os.mkdir(dataset_cache_dir)
        os.mkdir(os.path.join(dataset_cache_dir,'inv_hvp'))



    # Load the jedi dataset.
    dataset = JediDataset(features_path=features_path, name=dataset_name)
    
    # Add intercept to train and test data.
    dataset.train_X = np.concatenate((np.ones((dataset.train_X.shape[0], 1)), dataset.train_X), axis=1)
    dataset.test_X = np.concatenate((np.ones((dataset.test_X.shape[0], 1)), dataset.test_X), axis=1)

    # num features/dimensions
    nD = dataset.train_X.shape[1]

    # Compute the logistic regression model. Ignore appending intercept to the data. Load from cache if the model already exists.
    model_path = os.path.join(dataset_cache_dir, 'model_{}_{}.dat'.format(dataset_name, loss_type))
    if not force_refresh and os.path.exists(model_path):
        logger.info('Loading the model from the cache file - {}'.format(model_path))
        model = joblib.load(model_path)
    else:
        logger.info('Generating the model and saving it to cache - {}'.format(model_path))
        model = LR_UnbiasedEstimator(loss_type=loss_type, fit_intercept=False)
        model.fit(dataset.train_X, dataset.train_Y)
        joblib.dump(model, model_path, compress=3)

    # Load the model coefficients.
    W = model.coefficients()
    assert W.shape[0] == nD

    # populate the test labels using the logistic regression model.
    # The value is continuous in the interval [-1,1]
    pY = model.predict_prob(dataset.test_X)
    dataset.test_Y = (pY*2.) - 1.

    tf.reset_default_graph()
    vW = tf.constant(W, name='w', dtype=tf.float32)
    influence = Influence(W=vW, loss_type=loss_type)

    num_train = dataset.train_X.shape[0]

    # compute class weights.
    class_weights = {}
    unique_classes = np.unique(dataset.train_Y, return_counts=True)
    for idx in range(unique_classes[0].shape[0]):
        c = unique_classes[0][idx]
        v = unique_classes[1][idx]/num_train
        class_weights[c] = v


    # Merge train and test data.
    X = np.vstack([dataset.train_X, dataset.test_X])
    Y = np.hstack([dataset.train_Y, dataset.test_Y])


    # Compute the marginal distance.
    marginal_distance = np.abs(np.dot(X,W))/np.linalg.norm(W, ord=2)
    marginal_distance_path = os.path.join(dataset_cache_dir, 'marginal_distance_{}_{}.dat'.format(dataset_name, loss_type))
    joblib.dump(marginal_distance, marginal_distance_path, compress=3)


    # Load data set mapping into a pandas data frame.
    mapping_file = os.path.join(os.getenv('DATA_DIR'),'animal_breed_sdm/nameMapping_fullInfo_flipped0.2.mat')
    data = sio.loadmat(mapping_file)['nameMapping']

    rows = []
    for d in data:
        rows.append([x[0] for x in d.tolist()])
    df = pd.DataFrame(rows, columns=['img_name', 'common_name', 'dataset', 'train_test', 'class', 'is_flipped'])
    all_names = dataset.train_file_names + dataset.test_file_names


    if test_mode:
        tr_pos = np.where(dataset.train_Y_ORIG > 0)[0][:20]
        tr_neg = np.where(dataset.train_Y_ORIG < 0)[0][:20]
        te_pos = np.where(dataset.test_Y_ORIG > 0)[0][:5]
        te_neg = np.where(dataset.test_Y_ORIG < 0)[0][:5]

        tr_idx = np.hstack([tr_pos, tr_neg])
        te_idx = np.hstack([te_pos, te_neg])

        tr_names = [dataset.train_file_names[i] for i in tr_idx]
        te_names = [dataset.test_file_names[i] for i in te_idx]

        names = tr_names + te_names

        te_idx = te_idx + num_train

        list_of_idx = np.hstack([tr_idx, te_idx])


        file_names = [x.split('.')[0] for x in names]
        df_names = pd.DataFrame(file_names, columns=['fname'])

        df_filt = pd.merge(left=df, right=df_names, left_on='common_name', right_on='fname', how='inner')

    else:
        list_of_idx = np.arange(X.shape[0])



    inf_pert_loss = np.zeros(shape=[X.shape[0], num_train])
    is_flipped = np.zeros(shape=X.shape[0])

    for idx in list_of_idx:

        # compute the gradient w.r.t the given example. Please note that the given example could be test or train point in our framework.
        x = X[idx,:]
        y = Y[idx]

        vX = tf.Variable(x, 'test_x', dtype=tf.float32)
        vY = tf.Variable(y, 'test_y', dtype=tf.float32)

        test_dl_dw = influence.dl_dw(vX, vY, vW)


        file_name = all_names[idx].split('.')[0]
        _df = df[df.common_name == file_name].values[0]
        if _df[5] == 'flipped':
            is_flipped[idx] = 1

        # compute the hvp (hessian vector product) using the gradient for the given example.
        cache_file = os.path.join(os.path.join(dataset_cache_dir,'inv_hvp','inv_hvp_{}_{}.npz'.format(loss_type, idx)))
        if not force_refresh and os.path.exists(cache_file):
            logger.debug('Loading HVP file for idx {} from cache at {}'.format(idx, cache_file))
            inv_hvp = np.load(cache_file)['inv_hvp']
        else:
            start = datetime.datetime.now()
            inv_hvp = influence.inv_hvp_lissa_fast(dataset, test_dl_dw)
            end = datetime.datetime.now()
            exec_time = (end-start).total_seconds()
            logger.debug('Saving HVP file for idx {} to cache at {}'.format(idx, cache_file))
            np.savez_compressed(cache_file, inv_hvp=inv_hvp)

        # compute the influence of each training example on the given example
        influence_on_training_points = []
        # compute the influence of each training points.
        for train_idx in range(num_train):
            trX, trY = dataset.fetch_train_instance(train_idx)

            vX = tf.Variable(trX, 'X', dtype=tf.float32)
            vY = tf.Variable(trY, 'Y', dtype=tf.float32)

            dl_dydw = influence.dl_dydw(vX, vY, vW).numpy()
            a = -1* np.tensordot(dl_dydw, inv_hvp, axes=1).flatten()[0]

            _influence = a*class_weights[int(trY)]

            influence_on_training_points.append(_influence)

            inf_pert_loss[idx, train_idx] = _influence

        # save the results to the disk.
        results_file = os.path.join(dataset_cache_dir, 'inf_scores', 'influence_scores_{}_{}_{}_{}.dat'.format(idx,dataset_name, loss_type, is_flipped[idx]))
        joblib.dump(inf_pert_loss[idx,:], results_file, compress=3)
        logger.debug('Saving the influence scores on all the trainign poitns for idx {} to {}'.format(idx, results_file))



    # save the results to the disk.
    results_file = os.path.join(dataset_results_dir, 'influence_scores_{}_{}.dat'.format(dataset_name, loss_type))
    joblib.dump(inf_pert_loss, results_file, compress=3)
    logger.debug('Saving the perturbation loss results to the disk at {}'.format(results_file))

    is_flipped_file = os.path.join(dataset_results_dir, 'example_flipped_{}_{}.dat'.format(dataset_name, loss_type))
    joblib.dump(is_flipped, is_flipped_file, compress=3)
    logger.debug('Saving the flipped data to the disk at {}'.format(is_flipped_file))


    if test_mode:
        img_path = os.path.join(os.getenv('DATA_DIR'),'animal_breed_sdm/data_dog_flipped0.2/all/')
        rows = 2
        columns = 5

        for i in range(len(list_of_idx)):

            idx = list_of_idx[i]
            inf_scores = inf_pert_loss[idx,:]

            n_images = 10

            bot_10_idx = np.argsort(inf_scores)[:n_images]
            top_10_idx = np.argsort(-inf_scores)[:n_images]
            rnd_10_idx = np.random.randint(0, inf_scores.size, n_images)

            file_name = file_names[i]
            _df = df[df.common_name == file_name].values[0]


            if _df[5] == 'flipped':
                results_pdf_file = os.path.join(dataset_results_dir, 'test_mode_results_{}_{}_flipped.pdf'.format(i, loss_type))
            else:
                results_pdf_file = os.path.join(dataset_results_dir, 'test_mode_results_{}_{}.pdf'.format(i, loss_type))

            pp = PdfPages(results_pdf_file)
            img = mpimg.imread(os.path.join(img_path, '{}.jpg'.format(file_name)))

            # Test image.
            fig = plt.figure()
            plt.imshow(img)
            plt.title('%s/%s/[Flip:%s]' % (_df[3], _df[4], _df[5]))
            pp.savefig(fig)


            # Test image.
            fig = plt.figure()
            plt.plot(np.arange(inf_scores.shape[0]), inf_scores)
            plt.ylim(-2.,2.)
            plt.title('%s/%s/[Flip:%s]' % (_df[3], _df[4], _df[5]))
            pp.savefig(fig)

            # Bottom.
            fig = plt.figure(figsize=(20, 10))
            for i in range(10):
                img_idx = bot_10_idx[i]
                _file_name = dataset.train_file_names[img_idx].split('.')[0]
                _df = df[df.common_name == _file_name].values[0]
                _inf_score = inf_scores[img_idx]

                img = mpimg.imread(os.path.join(img_path, '{}.jpg'.format(_file_name)))
                fig.add_subplot(rows, columns, i + 1)
                plt.imshow(img)
                plt.title('%s/%0.6f/[Flip:%s]' % (_df[4], _inf_score,_df[5]))
            pp.savefig(fig)

            # Top.
            fig = plt.figure(figsize=(20, 10))
            for i in range(10):
                img_idx = top_10_idx[i]
                _file_name = dataset.train_file_names[img_idx].split('.')[0]
                _df = df[df.common_name == _file_name].values[0]
                _inf_score = inf_scores[img_idx]

                img = mpimg.imread(os.path.join(img_path, '{}.jpg'.format(_file_name)))
                fig.add_subplot(rows, columns, i + 1)
                plt.imshow(img)
                plt.title('%s/%0.6f/[Flip:%s]' % (_df[4], _inf_score,_df[5]))
            pp.savefig(fig)

            # Random.
            fig = plt.figure(figsize=(20, 10))
            for i in range(10):
                img_idx = rnd_10_idx[i]
                _file_name = dataset.train_file_names[img_idx].split('.')[0]
                _df = df[df.common_name == _file_name].values[0]
                _inf_score = inf_scores[img_idx]

                img = mpimg.imread(os.path.join(img_path, '{}.jpg'.format(_file_name)))
                fig.add_subplot(rows, columns, i + 1)
                plt.imshow(img)
                plt.title('%s/%0.6f/[Flip:%s]' % (_df[4], _inf_score,_df[5]))
            pp.savefig(fig)


            print('-----------------------')

            pp.close()





    # for test_idx in tgt_indices:
    #     start = datetime.datetime.now()
    #     inv_hvp = influence.inv_hvp_lissa(dataset, v_idx=test_idx, v_type=v_type)
    #     end = datetime.datetime.now()
    #     exec_time = (end-start).total_seconds()
    #     print('===== Executed in  {:0.2f} seconds ====='.format(exec_time))
    #
    #     influence_on_training_points = []
    #     # compute the influence of each training points.
    #     for train_idx in range(num_train):
    #         tr_X, tr_Y = dataset.fetch_train_instance(train_idx)
    #
    #         v_X = tf.Variable(tr_X, 'X', dtype=tf.float32)
    #         v_Y = tf.Variable(tr_Y, 'Y', dtype=tf.float32)
    #
    #         dl_dw = influence.dl_dw(v_X, v_Y, v_W)
    #         dl_dydw = influence.dl_dydw(v_X, v_Y, v_W).numpy()
    #         a = -1* np.tensordot(dl_dydw, inv_hvp, axes=1).flatten()[0]
    #
    #
    #         _influence = a*class_weights[int(tr_Y)]
    #
    #         influence_on_training_points.append(_influence)
    #
    #         inf_pert_loss[test_idx, train_idx] = _influence
    #
    #     # np.savez_compressed('./cache/inf_of_test_{}_{}_{}_{}.npz'.format(v_type, dataset_name, loss_type, test_idx), influence = _influence)
    #
    # np.savez_compressed('./cache/{}/inf_pert_loss_{}_{}_{}.npz'.format(dataset_name, v_type, dataset_name, loss_type),
    #                     inf_pert_loss=inf_pert_loss)

    print('**************************** COMPLETE *****************************************')

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='Compute Influence.')
    parser.add_argument('-l','--loss', help='Loss function to be used', choices=['unbiased','logistic'], default='logistic')
    parser.add_argument('-t','--test-run', help='test run on few examples as a quick smoke test', action='store_true')
    parser.add_argument('-d','--dataset', help='dataset', choices=['cat','dog','text'], default='dog')
    parser.add_argument('-f','--force', help='force refresh', action='store_true')
    parser.add_argument('-v','--verbose', help='print debug messages', action='store_true')
    args = vars(parser.parse_args())

    # TODO: Move to a configuration file.
    feat_dirs = {
        'dog': 'animal_breed_sdm/data_dog_flipped0.2',
        'cat': 'animal_breed_sdm/data_cat_flipped0.2',
        'text': 'text',
    }

    dataset = args['dataset']
    data_dir = os.getenv('DATA_DIR')
    feat_path = os.path.join(data_dir, feat_dirs[dataset])
    test_mode = args['test_run']
    loss_type = args['loss']
    verbose = args['verbose']
    force_refresh = args['force']

    if force_refresh:
        print('-------------------------------------------------------------')
        print('                           REFRESH')
        print('-------------------------------------------------------------')

    if verbose or test_mode:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    main(feat_path, dataset, loss_type=loss_type, test_mode=test_mode, force_refresh=force_refresh)
