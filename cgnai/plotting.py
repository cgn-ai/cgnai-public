# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/05_plotting.ipynb.

# %% auto 0
__all__ = ['to_uint8', 'ani']

# %% ../notebooks/05_plotting.ipynb 2
import imageio
import numpy as np
import matplotlib.pyplot as plt


def to_uint8(X):
    if isinstance(X, list): X = np.array(X)
    Y = X - np.amin(X)
    Y = Y/np.amax(Y)
    Y = Y*256
    Y = Y.astype(np.uint8)
    return Y
    

def ani(fname, X, **kwargs):
    
    if isinstance(X, list) and not isinstance(X[0], str): 
        X = np.array(X)
        
    if isinstance(X, np.ndarray):
        if X[0].dtype != np.uint8: 
            X = to_uint8(X)
    
    with imageio.get_writer(fname, **kwargs) as writer:
                
        for x in X: 
            if isinstance(x, str): x = imageio.imread(x)
            writer.append_data(x)

