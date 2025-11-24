import numpy as np
import pandas as pd

class RandomClassDataloader:
    def __init__(self,X,z,  ratio_non_drifting=0.333,drifting_classes=1,non_drifting_classes=1):
        self.X = X
        self.z = z
        self.z_clazz, self.z_count = np.unique(z, return_counts=True)
        self.ratio_non_drifting,self.drifting_classes,self.non_drifting_classes = ratio_non_drifting,drifting_classes,non_drifting_classes
        self.ratio()
    def ratio(self,ratio_non_drifting=None,drifting_classes=None,non_drifting_classes=None):
        if ratio_non_drifting is not None:
            self.ratio_non_drifting = ratio_non_drifting
        if drifting_classes is not None:
            self.drifting_classes = drifting_classes
        if non_drifting_classes is not None:
            self.non_drifting_classes = non_drifting_classes
        
        drifting_classes,non_drifting_classes = self.drifting_classes,self.non_drifting_classes

        prop = np.zeros(shape=(3,self.z_clazz.shape[0]))
        
        sel = np.random.choice(prop.shape[1],size=2*drifting_classes+non_drifting_classes, p = self.z_count / self.z_count.sum(), replace=False)

        prop[0,sel[:drifting_classes]] = self.z_count[sel[:drifting_classes]] 
        prop[1,sel[drifting_classes:-drifting_classes]] = self.z_count[sel[drifting_classes:-drifting_classes]] 
        prop[2,sel[-drifting_classes:]] = self.z_count[sel[-drifting_classes:]] 
        
        prop = prop / prop.sum(axis=1)[:,None]

        self.prop = prop
        
        
    def take(self, size, create_testset=False):
        number_of_drifting = int((1-self.ratio_non_drifting)*size/2)
        number_of_non_drifting = size-2*number_of_drifting
        number_of_non_drifting_before = number_of_non_drifting//2
        number_of_non_drifting_after = number_of_non_drifting-number_of_non_drifting_before
        if np.random.random() > 0.5:
            number_of_non_drifting_before, number_of_non_drifting_after = number_of_non_drifting_after, number_of_non_drifting_before
        
        sel = [
            np.random.choice(range(self.prop.shape[1]),p=self.prop[0], size=number_of_drifting, replace=True),
            np.random.choice(range(self.prop.shape[1]),p=self.prop[1], size=number_of_non_drifting, replace=True),
            np.random.choice(range(self.prop.shape[1]),p=self.prop[2], size=number_of_drifting, replace=True)]
        
        typ = [-np.ones(shape=number_of_drifting),
               -0.25*np.ones(shape=number_of_non_drifting_before),
               0.25*np.ones(shape=number_of_non_drifting_after),
               np.ones(shape=number_of_drifting)]
        perm = np.random.permutation(size)
        sel,typ = np.hstack(sel)[perm], np.hstack(typ)[perm]

        X = np.empty( shape=(size, self.X.shape[1]) )
        cls,cls_c = np.unique(sel,return_counts=True)
        for tp,n in zip(cls,cls_c):
            X[sel==tp] = self.X[np.random.choice(np.where(self.z==tp)[0],size=n,replace=True)]

        observation_state = (typ < 0).astype(int)
        drift_state = (np.abs(typ) > 0.5).astype(int)
        z = (2*observation_state-1)*drift_state
        
        if create_testset:
            raise ValueError()
            return (X,observation_state,drift_state,z), ()

        return X,observation_state,drift_state,z

class NoImageNetLoader(RandomClassDataloader):
    def __init__(self,**kwds):
        df = np.load("data/NINCO_embedding.npz")
        super().__init__(df["X"],df["z"], **kwds)
class FashionMNISTLoader(RandomClassDataloader):
    def __init__(self,**kwds):
        df = np.load("data/FashionMNIST.npz")
        super().__init__(df["X"].reshape(df["y"].shape[0],-1),df["y"], **kwds)

class FishHeadLoader:
    def __init__(self,ratio_non_drifting=0.333):
        self.df = np.load("data/fish_head_embedding.npz")
        self.q = ratio_non_drifting
    def ratio(self, ratio_non_drifting=None):
         if ratio_non_drifting is not None: 
             self.q = ratio_non_drifting
    def take(self, size, create_testset=False):
        embedding,z = self.df["X"], self.df["z"]-1
        q = 1-self.q
        sel = np.hstack( [np.random.choice(np.where(z==i)[0], size=int( size *(q/2 if i != 0 else 1-q)), replace=True) for i in np.unique(z)] ).flatten()
        train_observation_state = np.zeros(sel.shape[0],dtype=int)
        train_observation_state[z[sel]==1] = 1
        train_observation_state[np.random.choice(np.where(z[sel]==0)[0],size=(z[sel]==0).sum()//2,replace=False)] = 1

        if create_testset:
            nsel = np.zeros(embedding.shape[0], dtype=bool)
            nsel[sel] = False
            return (embedding[sel],train_observation_state,np.abs(z[sel]),z[sel]), (embedding[nsel],z[nsel])

        return embedding[sel],train_observation_state,np.abs(z[sel]),z[sel]

