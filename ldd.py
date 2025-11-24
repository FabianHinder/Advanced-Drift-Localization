from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.stats as stats
import numpy as np

def validate_y(y):
    y = y.flatten()
    entries = np.unique(y)
    assert entries.shape[0] == 2
    y_ = np.empty(y.shape)
    y_[y==entries[0]] = 0
    y_[y==entries[1]] = 1
    return y_


def LDD_delta(X,y,k):
    proba = KNeighborsClassifier(n_neighbors=k).fit(X,y).predict_proba(X)[np.arange(y.shape[0]), y]
    return (1.-proba)/proba -1
class LDDLocalizer(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.05, k=0.1, localizer_model=KNeighborsRegressor(n_neighbors=1)):
        self.k = k
        self.localizer_model = localizer_model
        self.alpha=alpha
    def get_info(self):
        return {"alpha": self.alpha, "localizer_model": str(self.localizer_model), "k": self.k, "class": self.__class__.__name__}

    def fit(self, X, y):
        y = validate_y(y).astype(int)
        assert len(X.shape) == 2 and X.shape[0] == y.shape[0]
        
        if self.k < 1:
            k = int(y.shape[0]*self.k)
        else:
            k = int(self.k)
        
        delta_val = LDD_delta(X, y, k)
        std_delta = LDD_delta(X, np.random.permutation(y), k).std()
        
        p = stats.norm.cdf(delta_val, loc=0, scale=std_delta)
        
        self.localizer_model.fit(X, np.vstack( (p,1-p) ).min(axis=0))
        
        return self

    def predict(self, X):
        return self.score_samples(X) < self.alpha

    def score_samples(self, X):
        return self.localizer_model.predict(X)


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    import matplotlib.pyplot as plt
    
    X = np.random.random(size=(500,2))
    X[:250,:2] -= 0.5
    w = np.linspace(0,1,X.shape[0]) < 0.5 
    
    plt.title("Datapoint (color is bd/ad)")
    plt.scatter(X[:,0],X[:,1], c=w)
    plt.show()

    
    loc = LDDLocalizer()
    loc.fit(X,w)
    plt.title("Result of integrated localizer class LDDLocalizer")
    plt.scatter(X[:,0],X[:,1], c=loc.predict_proba(X))
    plt.show()