import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone, BaseEstimator, ClassifierMixin, RegressorMixin
from joblib import Parallel, delayed

from util import greedy_cover

def get_ib_bootstraps(population_size, oob_times, bootstrap_fraction=.1, n_repititions=50):
    S = np.random.choice(range(population_size), size=(n_repititions*oob_times,int(bootstrap_fraction*population_size)), replace=True)
    M = np.ones(shape=(S.shape[0],population_size))
    
    for i,s in enumerate(S):
        M[i][s] = 0
    
    s = greedy_cover(1-M, np.ones(M.shape[0]), oob_times)
    return S[s]

def evaluate_bootstrap(X, y, bootstrap, base_model, n_samples, min_test_size):
    p_i = np.nan*np.ones( shape=(n_samples) )
    msg = []
    
    train_index = np.unique(bootstrap)
    test_index = np.ones(shape=n_samples, dtype=bool)
    test_index[bootstrap] = False
    test_index = np.where(test_index)[0]

    if np.unique(y[test_index]).shape[0] < 2:
        msg.append("WARN too small test set, try reducing bootstrap fraction. Skip iteration")
    else:
        if np.unique(y[test_index], return_counts=True)[1].min() < min_test_size: 
            msg.append("WARN few test samples (%i), try reducing bootstrap fraction. Instable precition"%np.unique(y[test_index], return_counts=True)[1].min())
    
        ## CONFORMAL PREDICTION ON IB SMAPLES USING OOB AS CLIBRATION SET
        probs = clone(base_model).fit(X[bootstrap],y[bootstrap]).predict_proba(X)[:,1]
        pos_cls0 = test_index[y[test_index] == 0]
        pos_cls1 = test_index[y[test_index] == 1]
        p_i[train_index] = np.minimum(
            ((probs[train_index,None] <= probs[None,pos_cls0]).sum(axis=1)+1)/(pos_cls0.shape[0]+1),
            ((probs[train_index,None] >= probs[None,pos_cls1]).sum(axis=1)+1)/(pos_cls1.shape[0]+1) )
    return p_i, msg

class ConformalLocalizer(BaseEstimator, ClassifierMixin):
    def __init__(self, model, cv_params={}, cv_runs=5, localizer=KNeighborsRegressor(n_neighbors=1), n_min_members=100, bootstrap_fraction=1., cover_search=50, alpha=0.2, n_jobs=-1):
        assert hasattr(model, "predict_proba")
        
        self.model = model
        self.cv_params = cv_params
        self.cv_runs = cv_runs
        self.localizer = localizer
        self.localizer_trained = None
        self.n_min_members = n_min_members
        self.bootstrap_fraction = bootstrap_fraction
        self.cover_search = cover_search
        self.alpha=alpha
        self.n_jobs = n_jobs
        
    def get_info(self):
        return {"base model": str(self.model), "cv parameter": self.cv_params, "cv runs": self.cv_runs, "localizer model": str(self.localizer_model), 
                "min members": self.n_min_members, "bootstrap fraction": self.bootstrap_fraction, "cover_search": self.cover_search, 
                "alpha": self.alpha, "jobs": self.n_jobs, "call": str(self), "class": self.__class__.__name__}
    def fit(self, X, y):
        assert len(X.shape) == 2 and X.shape[1] > 0 and X.shape[0] > 0
        assert len(y.shape) == 1 and np.unique(y).shape[0] == 2
        assert X.shape[0] == y.shape[0]
        y = LabelEncoder().fit_transform(y)
        n_samples = X.shape[0]
        
        base_model = clone(self.model)
        base_model.set_params(**GridSearchCV(estimator=clone(self.model), param_grid=self.cv_params, cv=self.cv_runs, n_jobs=self.n_jobs).fit(X,y).best_params_)

        bootstraps = get_ib_bootstraps(n_samples, self.n_min_members, bootstrap_fraction=self.bootstrap_fraction, n_repititions=self.cover_search)

        p = np.empty( shape=(bootstraps.shape[0],n_samples) )
        msgs = []
        for i,(p_i,msg) in enumerate(Parallel(n_jobs=self.n_jobs)(delayed(evaluate_bootstrap)(X,y,bootstrap, base_model, n_samples, 1/self.alpha+1) for bootstrap in bootstraps)):
            msgs += msg
            p[i] = p_i
        
        if len(msgs) > 0:
            for msg in set(msgs):
                print(msg)
        
        p_sel = (~np.isnan(p)).sum(axis=0)
        if (p_sel<self.n_min_members).any():
            print("WARN: small ensamble size ",p_sel.min())
        self.localizer_trained = clone(self.localizer).fit(X[p_sel > 0],np.nanmedian(p[:,p_sel > 0], axis=0))
        return self
        
    def predict(self, X):
        return self.score_samples(X) < self.alpha
    def score_samples(self, X):
        if self.localizer_trained is None:
            raise ValueError("Not fitted")
        return self.localizer_trained.predict(X)
