import logging
from sklearn import linear_model
from sklearn.metrics import accuracy_score


class LogisticRegression():

    def __init__(self, dataset, loss_function, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.model = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
        self.dataset = dataset
        self.loss_function = loss_function

    
    def train(self):
        self.model.fit(self.dataset.train_X, self.dataset.train_Y)
        
        Y_pred = self.model.predict(self.dataset.train_X)
        self.logger.info('Training accuracy: {:0.3f}'.format(accuracy_score(Y_pred, self.dataset.train_Y)))

        Y_pred = self.model.predict(self.dataset.test_X)
        self.logger.info('Test accuracy {:0.3f}'.format(accuracy_score(Y_pred, self.dataset.test_Y)))

    
    def parameters(self):
        W = self.model.coef_.transpose() 
        self.logger.info('Returning the model coefficients W - {}'.format(W.shape))
        return W
