import numpy as np
from math import exp

class LR_UnbiasedEstimator(object):
    def __init__(self, lr=0.01, num_iter=1000, fit_intercept=True, verbose=False, lamb=0.01, loss_type = 'logistic', noise_rate=0.2):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.lamb = lamb # lambda -> regularization strength
        self.loss_type = loss_type # 0 is Logistic Loss and 1 is Unbiased loss with noise.
        self.noise_rate = noise_rate
        self.theta = None
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        if z >= 0:
            return 1 / (1 + exp(-z))
        else:
            return exp(z)/(1 + exp(z))
        
    def __loss(self, h, y, z):
        if self.loss_type == 'unbiased':
            # this is +1/-1 LR modified loss
            alpha = 1 - 2 * self.noise_rate
            return (1-self.noise_rate)/alpha * (np.log(1 + np.exp(-1 * z * y))).mean() \
                     + self.noise_rate/alpha * (np.log(1 + np.exp(z * y))).mean()
        elif 'logistic':
            # this is +1/-1 LR logistic loss
            return (np.log(1 + np.exp(-1 * z * y))).mean() #+ 0.5 * self.lamb * np.linalg.norm(self.theta, ord=2)**2
        else:
            raise Exception('Unknown loss function type {} encountered. Logistic and unbiased loss are supported.'.format(loss_type)) 
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.random.rand(X.shape[1]) - 0.5
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta) # z = w^T x

            if z.shape[0] == 1:
                z = z.transpose()
                z = z.reshape(-1)

            h = np.array(list(map(self.__sigmoid, z))) # h = sigmoid(z)
            
            if self.loss_type == 'logistic':  # this is +1/-1 LR logistic loss
                h_neg = np.array(list(map(self.__sigmoid, -1*y*z))) # h_neg = sigmoid(-1 * y * z) 
                gradient = -1 * np.dot(X.T, y * h_neg) / y.size #+ self.lamb * self.theta
            elif self.loss_type == 'unbiased': # this is +1/-1 LR modified loss
                alpha = 1 - 2 * self.noise_rate
                h_neg = np.array(list(map(self.__sigmoid, -1*y*z))) # h_neg = sigmoid(-1 * y * z) 
                h_pos = np.array(list(map(self.__sigmoid, y*z))) # h_pos = sigmoid(y * z) 
                gradient = (1-self.noise_rate)/alpha * (-1 * np.dot(X.T, y * h_neg)) / y.size \
                            +  self.noise_rate/alpha * (np.dot(X.T, y * h_pos)) / y.size

            else:
                raise Exception('Unknown loss type {} found. Only logistic and unbiased loss functions are supported.'.format(self.loss_type))
            
            self.theta -= self.lr * gradient
            z = np.dot(X, self.theta)
            h = np.array(list(map(self.__sigmoid, z)))
            loss = self.__loss(h, y, z)
                
            if(self.verbose ==True and i % 500 == 0):
                print(f'Loss of iter [{i}] is: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return np.array(list(map(self.__sigmoid, np.dot(X, self.theta))))
    
    def predict(self, X):
        return ((self.predict_prob(X) > 0.5) * 1 - 0.5) * 2

    def coefficients(self):
        return self.theta
