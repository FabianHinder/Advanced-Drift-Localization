from tqdm import tqdm
import pandas as pd
from time import time, sleep
import sys
from sklearn.base import clone
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from kdq_tree import kdqTreeLocalizer 
from ldd import LDDLocalizer
from model_based import ForestLocalizer, KNNLocalizer, RandomTreeLocalizer
from tree_localizer import TreeLocalizer
from conformal_localizer import ConformalLocalizer

from dataloader import NoImageNetLoader, FashionMNISTLoader, FishHeadLoader

assert len(sys.argv) in [1,2], str(list(sys.argv))
if len(sys.argv) == 2:
    param = int(sys.argv[1])
else:
    param = -1


res, t_save = [], time()
for split_id,(q,loader,size) in enumerate(tqdm(500*[(.8,FishHeadLoader(),500),(.9,NoImageNetLoader(),120),(.9,FashionMNISTLoader(),120)])):
    loader.ratio(ratio_non_drifting=q)
    X,os,ds,z = loader.take(size)

    for localizer in [ForestLocalizer(),LDDLocalizer(),kdqTreeLocalizer(only_before=False), 
          KNNLocalizer(),RandomTreeLocalizer(),TreeLocalizer(),
          ConformalLocalizer(model=DecisionTreeClassifier(), cv_params={"min_samples_leaf": [10,15,20,30,50,100]}),
          ConformalLocalizer(model=MLPClassifier(max_iter=1000), cv_params={'hidden_layer_sizes': [(10,), (50,), (100,)], 'alpha': [0.0001, 0.001, 0.01]})]:
        try:
            localizer = clone(localizer)
            
            t0 = time()
            localizer.fit(X,os)
            p = localizer.score_samples(X)
            t1 = time()
            
            assert (~np.isnan(p)).all()
            res.append({"param": param, "exp_id": split_id, 
                        "q": q, "size": size, "loader": str(loader), 
                        "localizer": str(localizer), "drifting samples": ds.sum(),
                        "score": roc_auc_score(ds,1-p), 
                        "time": t1-t0})
        except Exception as e:
            print(q,loader,localizer,size,e)
    if time() - t_save > 10*60:
        try:
            pd.DataFrame(res).to_pickle(f"out/res_{param}.pkl.xz")
            t_save = time()
        except Exception as e:
            print(e)
            pass

while True:
    try:
        pd.DataFrame(res).to_pickle(f"out/res_{param}.pkl.xz")
        break
    except Exception as e:
        print(e)
        sleep(10)
print("OK")
