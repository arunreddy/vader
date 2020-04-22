import sys
import os
import logging
import pandas as pd
import numpy as np
import pickle
from glob import glob
from sklearn.metrics import accuracy_score



class JediDataset(object):
    
    def __init__(self, name, features_path, noise_rate = 0.1):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.features_path = features_path
        self.dict_features = None
        
        self.label_map = {'domestic':1, 'wild':-1}
        
        self.train_X = None
        self.train_Y = None
        self.train_Y_ORIG = None

        self.test_X = None
        self.test_Y = None
        self.test_Y_ORIG = None
        
        self.train_file_names = []
        self.test_file_names = []

        self.noise_rate  = noise_rate
        
        if os.path.exists(self.features_path):
            
            if self.name == 'dog' or self.name == 'cat':
                self.load_cats_dogs_from_pkl()
            elif self.name == 'text':
                self.load_text_dataset_from_pkl()
            
            
        else:
            raise RuntimeError('The provided path to the features doesn\'t exist.')
        
    def fetch_train_batch(self, batch_size):

        n_train = self.train_X.shape[0]
        assert batch_size <= self.train_X.shape[0]
        
        # Shuffle the data
        idx = np.arange(n_train)
        np.random.shuffle(idx)
        
        batch_idx = idx[:batch_size]
        
        _train_X = self.train_X[batch_idx,:].copy()
        _train_Y = self.train_Y[batch_idx].copy()
        
        return _train_X, _train_Y
        
    def fetch_test_instance(self, idx):
        test_X = self.test_X[idx,:]
        if len(test_X.shape) < 2:
            test_X = test_X.reshape(-1,1).transpose()
        
        # return a matrix of the size 1xNUM_FEATURES
        assert test_X.shape[0] == 1
        return test_X, self.test_Y[idx]

    def fetch_train_instance(self, idx):
        train_X = self.train_X[idx,:]
        if len(train_X.shape) < 2:
            train_X = train_X.reshape(-1,1).transpose()
        
        # return a matrix of the size 1xNUM_FEATURES
        assert train_X.shape[0] == 1
        return train_X, self.train_Y[idx]

    
    def train_accuracy_score(self,  train_Y):
        return accuracy_score(self.train_Y_ORIG, train_Y)


    def load_cats_dogs_from_pkl(self):
        self.logger.info('Loading dataset from the pickle file - {}'.format(self.features_path))
        for pkl_file_path in glob(os.path.join(self.features_path,'**.pickle')):
            file_name_w_ext = os.path.basename(pkl_file_path)

            with open(pkl_file_path, 'rb') as f:
                data = pickle.load(f)
                df = pd.DataFrame.from_dict(data, orient='columns').transpose()
            
                y = None
                for k in self.label_map:
                    if k in file_name_w_ext:
                        y = df.shape[0]*[self.label_map[k]]
                
                names = list(df.index)
                
                X = df.values
                y = np.asarray(y)
                                
                if 'train' in file_name_w_ext:
                    self.train_file_names += names
                    if self.train_X is None:
                        self.train_X = X
                        self.train_Y = y
                    else:
                        self.train_X = np.vstack((self.train_X, X))
                        self.train_Y = np.hstack((self.train_Y, y))
                
                # file_name.contains('val') or file_name.contains('test'):
                else: 
                    self.test_file_names += names

                    if self.test_X is None:
                        self.test_X = X
                        self.test_Y = y
                    else:
                        self.test_X = np.vstack((self.test_X, X))
                        self.test_Y = np.hstack((self.test_Y, y))


        self.train_Y_ORIG = self.train_Y.copy()
        self.test_Y_ORIG = self.test_Y.copy()
        if self.noise_rate > 0:
            flip_idx = np.random.binomial(1, self.noise_rate, self.train_Y.size)
            flip_vec = (flip_idx * -1 + 0.5 ) * 2
            flip_vec = flip_vec.astype(int)
            self.train_Y = self.train_Y_ORIG * flip_vec

        
        self.logger.info('Loading dataset Train- {}{} and Test-{}{}'.format(self.train_X.shape,
                                                                          self.train_Y.shape,
                                                                          self.test_X.shape,
                                                                          self.test_Y.shape))
    
    
    def _fetch_pkl_file_as_dataframe(self, pkl_file_path):
        df = None
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
            print(data.shape)
            df = pd.DataFrame.from_dict(data, orient='columns').transpose()
        return df
        
    def _fetch_pkl_file_as_array(self, pkl_file_path):
        data = None
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        return data
        
    def load_text_dataset_from_pkl(self):
        self.logger.info('Loading dataset from the pickle file - {}'.format(self.features_path))

        self.train_X = self._fetch_pkl_file_as_array(os.path.join(self.features_path,'features_train_flipped0.2.pickle'))
        self.train_Y_ORIG = self._fetch_pkl_file_as_array(os.path.join(self.features_path,'labels_train_gt.pickle'))
        self.train_Y = self._fetch_pkl_file_as_array(os.path.join(self.features_path,'labels_train_flipped0.2.pickle'))
        self.test_X = self._fetch_pkl_file_as_array(os.path.join(self.features_path,'features_test_flipped0.2.pickle'))
        self.test_Y_ORIG = self._fetch_pkl_file_as_array(os.path.join(self.features_path,'labels_test_gt.pickle'))
        
        
        print('Loading dataset Train- {}{} and Test-{}{}'.format(self.train_X.shape,
                                                                          self.train_Y.shape,
                                                                          self.test_X.shape,
                                                                          self.test_Y_ORIG.shape))
              
        