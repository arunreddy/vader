'''
Adapted from the following:
 1. https://github.com/kohpangwei/influence-release
 2. https://github.com/darkonhub/darkon/tree/master/darkon/influence
 3. https://github.com/HIPS/autograd
'''
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product


class Influence(object):
    
    def __init__(self, **kwargs):
        
        # set up default params.
        
        
        self.params = {}
        self.params['hvp_iterations'] = kwargs.get('hvp_iterations', 1)
        self.params['hvp_recursion_iterations'] = kwargs.get('hvp_recursion_iterations', 1000)
        self.params['hvp_scale'] = kwargs.get('hvp_scale', 1e4)
        self.params['hvp_decay'] = kwargs.get('hvp_decay', 0.01)
        self.params['hvp_batch_size'] = kwargs.get('hvp_batch_size', 10)
        
    
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


    def logistic_loss(self, x,y,w):
        z1 = tf.math.multiply(y,tf.tensordot(w,x, axes=1))
        z2 = tf.math.exp(-z1)
        z3 = tf.math.log(1 + z2)
        return z3
    
    
    def dl_dw(self, x,y,w):
        with tf.GradientTape(persistent=True) as t:
            t.reset()
            l = self.logistic_loss(x, y, w)
        dl_dw = t.gradient(l, w)
        return dl_dw

    def dl_dydw(self, x,y,w):
        '''Second order graident of the loss function w.r.t the parameter w(theta) and the label y'''
        with tf.GradientTape(persistent=True) as t1:
            t1.reset()
            t1.watch(y)
            t1.watch(x)
            with tf.GradientTape(persistent=True) as t:
                t.reset()
                t.watch(y)
                t.watch(x)
                l = self.logistic_loss(x, y, w)
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
            t1.watch(y)
            t1.watch(x)
            with tf.GradientTape(persistent=True) as t:
                t.reset()
                t.watch(y)
                t.watch(x)
                l = self.logistic_loss(x, y, w)
            dl_dw = t.gradient(l, w)
        dl_d2w = t1.gradient(dl_dw, w)
        return dl_d2w

    def grad_loss(self, l, x, y, w):
        '''Gradient of logistic loss term.'''
        with tf.GradientTape(persistent=True) as t:
            t1.reset()
        dl_dw = t.gradient(l, w)
        return dl_dw
    
    
    def inv_hvp_lissa(self, dataset, test_dl_dw):
        
        inv_hvp = None
        
        for _ in range(self.params['hvp_iterations']):
            cur_estimate = test_dl_dw
            
            for j in range(self.params['hvp_recursion_iterations']):

                train_X, train_Y = dataset.fetch_train_batch(self.params['hvp_batch_size'])
                
    
    
    
    
    def hvp_lissa(self, x, y, w, v):
        
        
        
        with tf.GradientTape(persistent=True) as t1:
            t1.reset()
            
            with tf.GradientTape(persistent=True) as t2:
                t2.reset()
                l = self.logistic_loss(x, y, w)
            
            # first backpropagation.
            dl_dw = t2.gradient(l, w)
            dl_dw = tf.multiply(w,x)
        
        print(dl_dw.shape)
        dl_d2w = t1.gradient(dl_dw, w)
        
        return dl_d2w