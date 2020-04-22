'''
Adapted from the following:
 1. https://github.com/kohpangwei/influence-release
 2. https://github.com/darkonhub/darkon/tree/master/darkon/influence
 3. https://github.com/HIPS/autograd
'''
import os
import pandas as pd
import numpy as np
import logging 
import matplotlib.pyplot as plt
import tensorflow as tf


class Influence(object):
    
    def __init__(self, W, **kwargs):

        print(kwargs)
        
        # set up default params.
        self.W = W
        self.logger = logging.getLogger(__name__)        
        self.params = {}
        self.params['hvp_iterations'] = kwargs.get('hvp_iterations', 1)
        self.params['hvp_recursion_iterations'] = kwargs.get('hvp_recursion_iterations', 2000)
        self.params['hvp_scale'] = kwargs.get('hvp_scale', 1e4)
        self.params['hvp_damping_factor'] = kwargs.get('hvp_damping_factor', 0.01)
        self.params['hvp_batch_size'] = kwargs.get('hvp_batch_size', 10)
        self.debug = kwargs.get('DEBUG', False)

        self.loss_type = kwargs.get('loss_type', 'logistic')
        self.logger.info('Initialized Influence object with the parameters: {}'.format(kwargs))

        if self.debug:
            print('*******************************************')
        
    
    def upweight_training_point(self):
        pass
    
    
    def influence_from_perturbation(self):
        pass
    
    def cross_entropy_loss(self, logits,labels):
        num_classes = logits.shape[-1]
        labels = tf.one_hot(labels, num_classes, axis=-1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels,
                                                                name='cross_entropy_per_example')

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def loss(self, x, y, w):

        if self.loss_type == 'logistic':
            return self.logistic_loss(x, y, w)
        elif self.loss_type == 'unbiased':
            # add the ability to read alpha, p_negy and p_posy from the parameters

            return self.surrogate_loss(x, y, w)
        else:
            raise Exception('Unknown loss type {} found. Only logistic and unbiased loss functions are supported.'.format(self.loss_type))


    def logistic_loss(self, x,y,w):
        z1 = tf.math.multiply(y,tf.tensordot(x,w, axes=1))
        z2 = tf.math.exp(-z1)
        z3 = tf.math.log(1 + z2)
        return z3
    
    def surrogate_loss(self, x, y, w, alpha=0.6, p_negy=0.2, p_posy=0.2):
        assert alpha > 0.

        a =  ( 1. - p_negy)/alpha
        b = p_posy/alpha

        return a * self.logistic_loss(x, y, w) + b * self.logistic_loss(x, -y, w)

    def dl_dw(self, x,y,w):
        with tf.GradientTape(persistent=True) as t:
            t.reset()
            t.watch(w)
            t.watch(x)
            t.watch(y)
            l = self.loss(x, y, w)
        dl_dw = t.gradient(l, w)
        return dl_dw

    def dl_dydw(self, x,y,w):
        '''Second order graident of the loss function w.r.t the parameter w(theta) and the label y'''
        with tf.GradientTape(persistent=True) as t1:
            t1.reset()
            t1.watch(y)
            t1.watch(x)
            t1.watch(w)
            with tf.GradientTape(persistent=True) as t:
                t.reset()
                t.watch(y)
                t.watch(x)
                t.watch(w)
                l = self.loss(x, y, w)
            dl_dw = t.gradient(l, w)
        dl_dydw = t1.gradient(dl_dw, y)

        # As y is a scalar, the above operation returns l1_norm(x)*dl_dydw
        # vectorize the scalar and return it.
        ret_val = dl_dydw*x/tf.reduce_sum(x)
        return ret_val

    def dl_d2w(self, x,y,w):
        '''Hessian of the loss function w.r.t the parameter w(theta)'''
        with tf.GradientTape(persistent=True) as t1:
            t1.reset()
            t1.watch(w)
            with tf.GradientTape(persistent=True) as t:
                t.reset()
                t.watch(y)
                t.watch(x)
                l = self.loss(x, y, w)
            dl_dw = t.gradient(l, w)
        dl_d2w = t1.gradient(dl_dw, w)
        return dl_d2w

    def grad_loss(self, l, x, y, w):
        '''Gradient of logistic loss term.'''
        with tf.GradientTape(persistent=True) as t:
            t.reset()
        dl_dw = t.gradient(l, w)
        return dl_dw


    def inv_hvp_lissa_fast(self,dataset, test_dl_dw):

        inv_hvp = None
        curr_hv = test_dl_dw
        debug_diffs = []
        for _ in range(self.params['hvp_iterations']):

            # if self.debug:
            #     debug_diffs_estimation = []
            #     prev_estimation_norm = np.linalg.norm(np.concatenate([a for a in curr_hv]))

            for j in range(self.params['hvp_recursion_iterations']):
                train_X, train_Y = dataset.fetch_train_batch(self.params['hvp_batch_size'])
                v_X = tf.Variable(train_X, dtype=tf.float32)
                v_Y = tf.Variable(train_Y, dtype=tf.float32)

                _hv = self.hvp_lissa(v_X, v_Y, self.W, curr_hv)
                curr_hv = test_dl_dw + (1 - self.params['hvp_damping_factor']) * curr_hv - _hv / self.params[
                    'hvp_scale']

                del v_X
                del v_Y

                # if self.debug and j%100 == 0:
                #     curr_estimation_norm = np.linalg.norm(np.concatenate([a for a in curr_hv]))
                #     debug_diffs_estimation.append(curr_estimation_norm - prev_estimation_norm)
                #     prev_estimation_norm = curr_estimation_norm

            # if self.debug:
            #     debug_diffs.append(debug_diffs_estimation)

        if inv_hvp is None:
            inv_hvp = np.array(curr_hv) / self.params['hvp_scale']

        return inv_hvp


    def inv_hvp_lissa(self, dataset, v_idx, v_type = 'test', force_reload=False):
        
        inv_hvp = None

        cache_file = os.path.join('cache',dataset.name, 'inv_hvp_{}_{}_{}_{}.npz'.format(dataset.name, self.loss_type, v_type, v_idx))
        if not force_reload and os.path.exists(cache_file):
            self.logger.debug('Returning the inv_hvp from the cache -> {}'.format(cache_file))
            return np.load(cache_file)['inv_hvp']

        # compute the gradient of loss for test instance.

        if v_type == 'train':
            v_data_X, v_data_Y = dataset.fetch_train_instance(v_idx)
        else:
            v_data_X, v_data_Y = dataset.fetch_test_instance(v_idx)
        
        v_X = tf.Variable(v_data_X, 'X', dtype=tf.float32)
        v_Y = tf.Variable(v_data_Y, 'Y', dtype=tf.float32)

        test_dl_dw = self.dl_dw(v_X, v_Y, self.W)
        self.logger.debug('Gradient of loss at the {} point with idx {} computed'.format(v_type, v_idx))

        curr_hv = test_dl_dw
        debug_diffs = []
        for _ in range(self.params['hvp_iterations']):
            
            # if self.debug:
            #     debug_diffs_estimation = []
            #     prev_estimation_norm = np.linalg.norm(np.concatenate([a for a in curr_hv]))
    
            for j in range(self.params['hvp_recursion_iterations']):

                train_X, train_Y = dataset.fetch_train_batch(self.params['hvp_batch_size'])
                v_X = tf.Variable(train_X, dtype=tf.float32)
                v_Y = tf.Variable(train_Y, dtype=tf.float32)

                _hv = self.hvp_lissa(v_X, v_Y, self.W, curr_hv)
                curr_hv = test_dl_dw + (1 - self.params['hvp_damping_factor']) * curr_hv - _hv / self.params['hvp_scale']

                del v_X
                del v_Y

                # if self.debug and j%100 == 0:
                #     curr_estimation_norm = np.linalg.norm(np.concatenate([a for a in curr_hv]))
                #     debug_diffs_estimation.append(curr_estimation_norm - prev_estimation_norm)
                #     prev_estimation_norm = curr_estimation_norm


            # if self.debug:
            #     debug_diffs.append(debug_diffs_estimation)

        if inv_hvp is None:
            inv_hvp = np.array(curr_hv) / self.params['hvp_scale']

        if self.debug:
            debug_diffs = np.asarray(debug_diffs).transpose().flatten()
            x = np.arange(debug_diffs.shape[0])
            plt.plot(x,debug_diffs)
            plt.show()
        
        # save the inverse_hvp to cache.
        np.savez_compressed(cache_file, inv_hvp=inv_hvp)

        return inv_hvp
    
    def hvp_lissa(self, x, y, w, v):
        
        n_examples = tf.constant(x.shape[0], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(w)
            t1.watch(v)
            
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(w)
                t2.watch(v)
                l = self.loss(x, y, w)
            
            # first backpropagation.
            dl_dw = t2.gradient(l, w)

            # elementwise multiplication of result of first backprop and v.
            dl_dw = tf.multiply(dl_dw,v)
        
        # second backpropagation
        dl_d2w = t1.gradient(dl_dw, w)
        dl_d2w = tf.div(dl_d2w, n_examples)
        
        return dl_d2w