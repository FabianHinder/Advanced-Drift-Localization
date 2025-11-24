from sklearn.neighbors import KNeighborsRegressor
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

def fit_(X,perm,start_node,start_dim,min_samps):
    do_split,dim,point,offset = find_dim_(X,perm,start_dim,min_samps)
    if do_split:
        sel = X[:,dim]<point
        X1,X2 = X[sel],X[~sel]
        left  = fit_(X1,perm,start_node+1,(start_dim+offset+1)%perm.shape[0],min_samps)
        right = fit_(X2,perm,start_node+1+len(left),(start_dim+offset+1)%perm.shape[0],min_samps)
        return [(False,dim,point,start_node+1,start_node+1+len(left))]+left+right
    else:
        return [(True,int(1e5),np.nan,-1,-1)]


def find_dim_(X, perm, start_dim, min_samps):
    for i in range(perm.shape[0]):
        dim = perm[(start_dim+i) % perm.shape[0]]
        dim_ok, split = check_dim_(X[:,dim], min_samps)
        if dim_ok:
            return True, dim,split, i
    else:
        return False, None,None, None

def check_dim_(Xi, min_samps):
    if Xi.shape[0] > min_samps:
        mi,ma = Xi.min(),Xi.max()
        mid = (mi+ma)/2
        if (Xi < mid).sum() > min_samps and (Xi >= mid).sum() > min_samps:
            return True,mid
    return False,None

def apply_(X, sel, res, node, is_leaf, split_dim, split_point, tree_left, tree_right):
    if is_leaf[node]:
        res[sel] = node
    else:
        subsel = X[sel][:,split_dim[node]] < split_point[node]
        apply_(X, sel[ subsel], res, tree_left[node], is_leaf, split_dim, split_point, tree_left, tree_right)
        apply_(X, sel[~subsel], res, tree_right[node], is_leaf, split_dim, split_point, tree_left, tree_right)

class kdqTree:
    def __init__(self,min_samps=10):
        self.min_samps = min_samps
        self.fitted_ = False
    def get_info(self):
        return {"min_samps": self.min_samps, "class": self.__class__.__name__}
    def fit(self, X, y=None):
        assert len(X.shape) == 2
        
        tree = fit_(X,np.random.permutation(X.shape[1]),0,0,self.min_samps)
        self.is_leaf     = np.array([x for x,_,_,_,_ in tree],dtype=bool)
        self.split_dim   = np.array([x for _,x,_,_,_ in tree],dtype=int)
        self.split_point = np.array([x for _,_,x,_,_ in tree],dtype=float)
        self.tree_left   = np.array([x for _,_,_,x,_ in tree],dtype=int)
        self.tree_right  = np.array([x for _,_,_,_,x in tree],dtype=int)
        self.fitted_ = True
        
        return self
        
    def apply(self, X):
        assert len(X.shape) == 2
        assert self.fitted_
        
        result = np.empty(X.shape[0])
        apply_(X, np.arange(X.shape[0]), result, 0, self.is_leaf, self.split_dim, self.split_point, self.tree_left, self.tree_right)
        return result
    
class kdqTreeLocalizer(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.05, min_samps=10, only_before=True, do_bootstrap_test = True, bootstraps=2500):
        self.min_samps = min_samps
        self.only_before = only_before
        self.do_bootstrap_test = do_bootstrap_test
        self.bootstraps = bootstraps
        self.alpha = alpha
        self.p = None
        self.fitted_ = False
    def get_info(self):
        return {"min_samps":self.min_samps, "only_before": self.only_before, "do_bootstrap_test": self.do_bootstrap_test, "bootstraps": self.bootstraps, "alpha": self.alpha, "class": self.__class__.__name__}
    def fit(self, X, y):
        y = validate_y(y)
        assert len(X.shape) == 2 and X.shape[0] == y.shape[0]
        
        self.tree = kdqTree(self.min_samps).fit(X if not self.only_before else X[y==0])
        leaf = self.tree.apply(X)
        
        y0, y1 = (y==0).sum(), (y==1).sum()
        
        self.d_kl_v = dict()
        if self.do_bootstrap_test:
            self.p = dict()
            y_boot = np.random.choice([0,1], size=(self.bootstraps,y.shape[0]), p=np.array([y0,y1])/y.shape[0], replace=True)
            y0_boot = (y_boot == 0).sum(axis=1)
            y1_boot = (y_boot == 1).sum(axis=1)
        else:
            self.p = None
        for l in np.unique(leaf):
            sel = leaf == l
            p = (y[sel] == 0).sum() / y0
            q = (y[sel] == 1).sum() / y1
            self.d_kl_v[l] = (p-q)*np.log( (p*(1-q))/(q*(1-p)) ) if (1-p)*(1-q)*p*q != 0 else 1e6
            
            if self.do_bootstrap_test:
                p_boot = (y_boot[:,sel] == 0).sum(axis=1) / y0_boot
                q_boot = (y_boot[:,sel] == 1).sum(axis=1) / y1_boot
                in_log = (1-p_boot)*(1-q_boot)*p_boot*q_boot
                out_log = p_boot-q_boot
                d_kl_v_boot = 1e6 * np.ones(self.bootstraps)
                d_kl_v_boot[in_log!=0] = out_log[in_log!=0] * np.log(in_log[in_log!=0])
                self.p[l] = (self.d_kl_v[l]  < d_kl_v_boot).mean()
        self.fitted_ = True
        return self
    
    def predict(self, X):
        return self.score_samples(X) < self.alpha
    
    def predict_proba(self, X):
        assert self.fitted_
        p = np.array(list(map(lambda l: self.p[l],self.tree.apply(X))))
        return np.vstack( (p,np.ones(p.shape)-p))
    
    def score_samples(self, X):
        assert self.fitted_
        return np.array(list(map(lambda l: self.d_kl_v[l],self.tree.apply(X))))
        
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    import matplotlib.pyplot as plt
    
    X = np.random.random(size=(500,2))
    X[:250,:2] -= 0.5
    w = np.linspace(0,1,X.shape[0]) < 0.5 
    
    plt.title("Datapoint (color is bd/ad)")
    plt.scatter(X[:,0],X[:,1], c=w)
    plt.show()

    loc = kdqTreeLocalizer()
    loc.fit(X,w)
    plt.title("Result of integrated localizer class kdqTreeLocalizer")
    plt.scatter(X[:,0],X[:,1], c=loc.predict_proba(X))
    plt.show()
    
    loc = kdqTreeLocalizer(only_before=False)
    loc.fit(X,w)
    plt.title("Result of integrated localizer class kdqTreeLocalizer")
    plt.scatter(X[:,0],X[:,1], c=loc.predict_proba(X))
    plt.show()