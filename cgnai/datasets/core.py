# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/nlq/datasets/00_core.ipynb.

# %% auto 0
__all__ = ['rslice']

# %% ../notebooks/nlq/datasets/00_core.ipynb 3
import numpy as np
from itertools import islice 

# %% ../notebooks/nlq/datasets/00_core.ipynb 4
def rslice(I, n):
    a = np.random.randint(len(I)-n)
    return islice(I,a,a+n)