import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


class MyModel():
    def __init__(self):
        self.model = Ridge()
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_stat_abs_deviation(self, X, y):
        y_pred = self.predict(X)
        difference_abs = np.abs(y_pred - y)
        
        return np.max(difference_abs), np.min(difference_abs), np.mean(difference_abs)

    def wrap_model(self):
        return {"w": self.model.coef_, "b": self.model.intercept_}
