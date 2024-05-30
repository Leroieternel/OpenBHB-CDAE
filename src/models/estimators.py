import numpy as np
import multiprocessing
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class AgeEstimator(BaseEstimator):
    """ Define the age estimator on latent space network features.
    """
    def __init__(self):
        n_jobs = multiprocessing.cpu_count()
        self.age_estimator = GridSearchCV(
            Ridge(), param_grid={"alpha": 10.**np.arange(-2, 3)}, cv=5,
            scoring="r2", n_jobs=n_jobs)

    def fit(self, X, y):
        self.age_estimator.fit(X, y)
        return self.score(X, y)

    def predict(self, X):
        y_pred = self.age_estimator.predict(X)
        return y_pred
    
    def score(self, X, y):
        prediction = cross_val_predict(self.age_estimator.best_estimator_, X, y, cv=5)
        print('ground truth age: ', y)
        print('predicted age: ', prediction)
        y_pred = self.age_estimator.predict(X)
        return mean_absolute_error(y, y_pred)

class SiteEstimator(BaseEstimator):
    """ Define the site estimator on latent space network features.
    """
    def __init__(self):
        n_jobs = multiprocessing.cpu_count()
        self.site_estimator = GridSearchCV(
            LogisticRegression(solver="saga", max_iter=150), cv=5,
            param_grid={"C": 10.**np.arange(-2, 3)},
            scoring="balanced_accuracy", n_jobs=n_jobs)

    def fit(self, X, y):
        self.site_estimator.fit(X, y)
        return self.site_estimator.score(X, y)

    def predict(self, X):
        return self.site_estimator.predict(X)

    def score(self, X, y):
        prediction = cross_val_predict(self.site_estimator.best_estimator_, X, y, cv=5)
        print('ground truth site: ', y)
        print('predicted site: ', prediction)
        cm = confusion_matrix(y, prediction)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # plt.show()
        plt.savefig('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/confusion_matrix_1.jpg')
        return self.site_estimator.score(X, y)
    
    # def plot_confusion_matrix(self, X, y):
    #     prediction = cross_val_predict(self.site_estimator.best_estimator_, X, y, cv=5)
        