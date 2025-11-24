from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.model_selection import cross_validate
import numpy as np
import copy

def validate_y(y):
    y = y.flatten()
    entries = np.unique(y)
    assert entries.shape[0] == 2
    y_ = np.empty(y.shape)
    y_[y==entries[0]] = 0
    y_[y==entries[1]] = 1
    return y_

def complement(sel, n):
    mask = np.ones(n, dtype=bool)
    mask[sel] = False
    return np.flatnonzero(mask)

## Dataset size N, before drift ratio p, sample size n, observe before drift count 
def count_to_probability(count, N, h0, n, lambert):
    assert 0 <= h0 and h0 <= 1
    assert (count.astype(int) == count).all()
    assert int(N) == N
    assert int(n) == n
    
    count, N, n = np.atleast_1d(count).astype(int), int(N), int(n)

    if not lambert:
         compute_probability = lambda distribution: 2*np.vstack((distribution.cdf(count),1-distribution.cdf(count-1))).min(axis=0)
    else:
        h = count / n
        h[h == 0.] = 0.5/n
        h[h == 1.] = 1-0.5/n
        h_lo, h_hi = compute_kl_interval(h, h0)
        if np.isnan(h_lo).any() or np.isnan(h_hi).any():
            print("NAN!", h0)
            print(h[np.logical_or(np.isnan(h_lo),np.isnan(h_hi))])
        count_lo, count_hi = np.floor(h_lo*n).astype(int), np.ceil(h_hi*n).astype(int)
        compute_probability = lambda distribution: distribution.cdf(count_lo)+(1-distribution.cdf(count_hi))
    
    true_dist = stats.hypergeom(N, int(N*h0), n)
    prob = compute_probability(true_dist)
    
    if np.isnan(prob).any():
        mu, sigma2 = true_dist.mean() / n, true_dist.var() / n**2
        if sigma2 > mu*(1-mu):
            print("WARNING ",mu,sigma2)
        a = mu*( mu*(1-mu)/sigma2 - 1)
        approx_dist = stats.beta(scale=n, a=a, b=a * (1/mu -1))
        prob = compute_probability(approx_dist)
    return prob

class ForestLocalizer(BaseEstimator, ClassifierMixin):
    def __init__(self, forest=RandomForestClassifier(n_estimators=100, n_jobs=-1), alpha=0.005, min_samples_leaf=None, lambert=False):
        self.forest = forest
        self.localizer_model = None
        self.mean = None
        self.k = None
        self.N = None
        self.alpha=alpha
        self.min_samples_leaf = min_samples_leaf
        self.lambert = lambert
    def get_info(self):
        return {"base_model": str(self.forest), "localizer_model": str(self.localizer_model), "alpha": self.alpha, "min_samples_leaf": self.min_samples_leaf, "lambert": self.lambert, "class": self.__class__.__name__}

    def fit(self, X, y):
        y = validate_y(y)
        self.N = y.shape[0]
        assert len(X.shape) == 2 and X.shape[0] == y.shape[0]
        
        if self.min_samples_leaf is None:
            gs = GridSearchCV(estimator=self.forest, param_grid={"min_samples_leaf": [10,15,20,30,50,100]}, cv=5, n_jobs=-1).fit(X,y)
            min_samples_leaf = gs.best_params_["min_samples_leaf"]
        else:
            min_samples_leaf = self.min_samples_leaf
        self.localizer_model = self.forest.set_params(min_samples_leaf=int(min_samples_leaf)).fit(X,y)
        self.mean = y.mean()
        self.k = int(np.median(list(map(lambda l: np.median(np.unique(l, return_counts=True)[1]), self.localizer_model.apply(X).T))))
        return self

    def predict(self, X):
        return self.score_samples(X) < self.alpha

    def score_samples(self, X):
        prob_pred = self.localizer_model.predict_proba(X)[:,0]
        pred_count = np.round(prob_pred*self.k).astype(int)
        return count_to_probability(pred_count, self.N, self.mean, self.k, self.lambert)

class KNNLocalizer(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.005, base_model=KNeighborsClassifier(), k_init=None, lambert=False):
        self.base_model = base_model
        self.localizer_model = None
        self.mean = None
        self.N = None
        self.k_init = k_init
        self.k = k_init if k_init is not None else 10
        self.alpha=alpha
        self.lambert = lambert
    def get_info(self):
        return {"base_model": str(self.base_model), "localizer_model": str(self.localizer_model), "alpha": self.alpha, "k": self.k_init, "lambert": self.lambert, "class": self.__class__.__name__}

    def fit(self, X, y):
        y = validate_y(y)
        self.N = y.shape[0]
        assert len(X.shape) == 2 and X.shape[0] == y.shape[0]
        
        if self.k_init is None:
            gs = GridSearchCV(estimator=self.base_model, param_grid={"n_neighbors": [k for k in [10,15,20,30,50,100] if k < X.shape[0]*(1-1/5)]}, cv=5, scoring='neg_log_loss', n_jobs=-1).fit(X,y)
            self.k = gs.best_params_["n_neighbors"]
        else:
            self.k = self.k_init
        self.localizer_model = self.base_model.set_params(n_neighbors=self.k).fit(X,y)
        self.mean = y.mean()
        return self

    def predict(self, X):
        return self.score_samples(X) < self.alpha

    def score_samples(self, X):
        prob_pred = self.localizer_model.predict_proba(X)[:,0]
        pred_count = np.round(prob_pred*self.k).astype(int)
        return count_to_probability(pred_count, self.N, self.mean, self.k, self.lambert)

class RandomTreeLocalizer(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.2, base_model=RandomTreesEmbedding(min_samples_leaf=10), localizer_model=KNeighborsRegressor(n_neighbors=1), lambert=False):
        self.base_model = base_model
        self.localizer_model = localizer_model
        self.alpha=alpha
        self.lambert = lambert
    def get_info(self):
        return {"base_model": str(self.base_model), "localizer_model": str(self.localizer_model), "alpha": self.alpha, "lambert": self.lambert, "class": self.__class__.__name__}

    def fit(self, X, y):
        y = validate_y(y)
        N = y.shape[0]
        assert len(X.shape) == 2 and X.shape[0] == y.shape[0]

        leavs = self.base_model.fit_transform(X).tocsc()
        
        mean = y.mean()
        count, prob = np.zeros(y.shape[0]), np.zeros(y.shape[0])
        for itr in range(leavs.shape[1]):
            sel = leavs[:,itr].indices
            count[sel] += 1
            prob[sel] += np.log(count_to_probability(y[sel].sum(), N, mean, sel.shape[0], self.lambert)+1e-64)
        
        prob[count == 0] = 0.5
        count[count == 0] = 1
        mean_prob = np.exp(2*prob/count)
        
        self.localizer_model.fit(X, mean_prob)
        
        return self

    def predict(self, X):
        return self.score_samples(X) < self.alpha

    def score_samples(self, X):
        return self.localizer_model.predict(X)

