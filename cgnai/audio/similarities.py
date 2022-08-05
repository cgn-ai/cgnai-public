# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/podverse/04_speaker_similarities.ipynb.

# %% auto 0
__all__ = ['logger', 'log', 'get_clusters', 'get_cluster_similarity', 'plot_super_similarity_matrix']

# %% ../notebooks/podverse/04_speaker_similarities.ipynb 2
import cgnai
from pathlib import Path
import sys
from ..logging import cgnai_logger
import numpy as np
from ..utils import cgnai_home
from ..fileio import ls, load
from .diarization import get_speaker_timeline, load_ids
from .embeddings import load_embedding
from .superpixels import load_super_pixels

logger = cgnai_logger("similarities")
log = logger.info

# %% ../notebooks/podverse/04_speaker_similarities.ipynb 9
def get_clusters(ids, I, min_ratio=0.1):
    v=get_speaker_timeline(ids, I)
    bins=np.linspace(-0.5, np.max(v)+ 0.5, np.max(v)+2)
    h=np.histogram(v, bins=bins)[0]
    ss = set(np.where(h > np.amax(h)*min_ratio)[0])
    cl = {}
    for i in ss:
        cl[i] = (v == i)
        
    return cl, v
    

# %% ../notebooks/podverse/04_speaker_similarities.ipynb 12
def get_cluster_similarity(cl_i, emb_i, cl_j, emb_j):
    T_i = len(emb_i)
    T_j = len(emb_j)

    n_i = len(cl_i.keys())
    n_j = len(cl_j.keys())
    
    csim = np.zeros((n_i, n_j))

    for ia,a in enumerate(cl_i.keys()):
        for ib,b in enumerate(cl_j.keys()):
            M = emb_i[cl_i[a],:]@emb_j[cl_j[b],:].T
            csim[ia,ib] = np.mean(M)
    return csim

# %% ../notebooks/podverse/04_speaker_similarities.ipynb 14
def plot_super_similarity_matrix(files):
    ids = {}
    Is = {}
    embs = {}
    clusters={}
    for fname in files:
        ids[fname] = load_ids(data_path / fname)
        Is[fname] = load_super_pixels(data_path / fname)
        embs[fname] = load_embedding(data_path / fname)
        clusters[fname], _ = get_clusters(ids[fname], Is[fname])
    log("Done loading")
        
    
    # Compute similarity matrices.
    csims = []
    for i in range(len(files)):
        csims.append([])
        for j in range(i, len(files)):
            print(f"{i} - {j}", end="\r")
            csim = get_cluster_similarity(clusters[files[i]], embs[files[i]], clusters[files[j]], embs[files[j]])
            csims[-1].append(csim)
    
    cum = [0, *np.cumsum([c.shape[1] for c in csims[0]])]
    super_sim = np.zeros((cum[-1],cum[-1]))

    n = len(csims)
    for i in range(n):
        for j_ in range(len(csims[i])):
            j = i + j_
            cij = csims[i][j_]

            super_sim[cum[i]:cum[i+1],cum[j]:cum[j+1] ] = cij
            super_sim[cum[j]:cum[j+1],cum[i]:cum[i+1] ] = cij.T
    
    plt.figure(figsize=(20,20))
    plt.imshow(super_sim, vmin=0.15, vmax=0.5)
    for c in cum:
        plt.hlines(c - 0.5,-0.5,cum[-1]-0.5, color="w", linewidth=2)
        plt.vlines(c - 0.5,-.5,cum[-1]-0.5, color="w", linewidth=2)
