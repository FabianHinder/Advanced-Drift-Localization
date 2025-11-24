import numpy as np

def greedy_cover(M, w, d, selected = None):
    M = M != 0
    n, m = M.shape  
    r = np.zeros(m)
    
    if selected is None:
        selected = np.zeros(n, bool)  
    assert selected.shape == (n,)

    while r.min() < d and not selected.all():
        gain = M[:, r < d].sum(axis=1) / w
        gain[selected] = -1
        i = gain.argmax()
        r[M[i]] += 1
        selected[i] = True
    return selected