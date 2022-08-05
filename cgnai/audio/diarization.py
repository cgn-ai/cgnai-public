# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/podverse/03_speaker_diarization.ipynb.

# %% auto 0
__all__ = ['logger', 'log', 'get_superpixel_sim_matrix', 'inflate_superpixel_sim_matrix', 'remap_ids', 'optimize_labels',
           'make_speaker_map', 'reconstruct_sim', 'get_speaker_timeline', 'load_ids']

# %% ../notebooks/podverse/03_speaker_diarization.ipynb 2
import cgnai
from pathlib import Path
import sys
from ..logging import cgnai_logger
import numpy as np
from ..fileio import ls, load

logger = cgnai_logger("diarization")
log = logger.info

# %% ../notebooks/podverse/03_speaker_diarization.ipynb 7
def get_superpixel_sim_matrix(d, I):
    N = len(I) - 1
    M = np.zeros((N, N))
    for i in range (0,N):
        for j in range(i, N):
            M[i,j] = M[j,i] = np.mean(d[I[i]:I[i+1],I[j]:I[j+1]])
    return M

# %% ../notebooks/podverse/03_speaker_diarization.ipynb 9
def inflate_superpixel_sim_matrix(M, I):
    T = I[-1]
    N = len(I) - 1
    d = np.zeros((T, T))
    for i in range (0,N):
        for j in range(i, N):
            d[I[i]:I[i+1], I[j]:I[j+1]] = M[i,j]
            d[I[j]:I[j+1], I[i]:I[i+1]] = M[i,j]
    return d

# %% ../notebooks/podverse/03_speaker_diarization.ipynb 12
def remap_ids(ids):
    unique_ids = list(ids[np.sort(np.unique(ids, return_index=True)[1])])
    return np.array([unique_ids.index(id) for id in ids ])

# %% ../notebooks/podverse/03_speaker_diarization.ipynb 13
import math

def optimize_labels(M, I, max_speaker=6, mu_same=0.55, mu_diff=0.15, sigma=0.1):
    N = len(I) - 1
    ids = np.random.randint(0, max_speaker, N)
    
    # precompute sigmas
    sigma_sq = np.zeros((N, N))
    for i in range(0, N):
        l_i = I[i+1] - I[i]
        for j in range(i, N):
            l_j = I[j+1] - I[j]
            sigma_sq[i, j] = sigma_sq[j, i] = sigma * sigma / (math.sqrt(l_i) * math.sqrt(l_j))
    
    n_updates = 1
    while n_updates > 0:
        n_updates = 0
        for i in range(0, N):
            log_ps = []
            for new_id_i in range(0, max_speaker):
                log_p = 0
                for j in range(0, N):
                    l_j = I[j+1] - I[j] # size of ith super pixel
                    mu = mu_same if new_id_i == ids[j] else mu_diff
                    if i == j:
                        mu = mu_same
                    delta = M[i, j] - mu
                    log_p += delta * delta / sigma_sq[i, j]
                log_ps.append(log_p)
            assert len(log_ps) == max_speaker
            new_id_i = np.argmin(log_ps)
            if new_id_i != ids[i]:
                ids[i] = new_id_i
                n_updates += 1
        
    log_p = 0
    for i in range(0, N):
        l_i = I[i+1] - I[i] # size of ith super pixel
        for j in range(0, N):
            l_j = I[j+1] - I[j] # size of ith super pixel
            mu = mu_same if ids[i] == ids[j] else mu_diff
            if i == j:
                mu = mu_same
            delta = M[i, j] - mu
            log_p += delta * delta / sigma_sq[i, j]
    return remap_ids(ids), log_p

# %% ../notebooks/podverse/03_speaker_diarization.ipynb 14
def make_speaker_map(I, ids):
    T = I[-1]
    N = len(I) - 1
    d = np.zeros((T, T))
    for i in range (0,N):
        for j in range(i, N):
            c = (ids[i] == ids[j]) * (1 + ids[i])
            d[I[i]:I[i+1], I[j]:I[j+1]] = c
            d[I[j]:I[j+1], I[i]:I[i+1]] = c
    return d

# %% ../notebooks/podverse/03_speaker_diarization.ipynb 15
def reconstruct_sim(I, ids, mu_same=0.5, mu_diff=0.15):
    T = I[-1]
    N = len(I) - 1
    d = np.zeros((T, T))
    for i in range (0,N):
        for j in range(i, N):
            c = mu_same if ids[i] == ids[j] else mu_diff
            d[I[i]:I[i+1], I[j]:I[j+1]] = c
            d[I[j]:I[j+1], I[i]:I[i+1]] = c
    return d

# %% ../notebooks/podverse/03_speaker_diarization.ipynb 17
def get_speaker_timeline(ids, I):
    T = I[-1]
    N = len(I)-1
    timeline = np.zeros((T))
    for i in range(0, N):
        timeline[I[i]:I[i+1]] = ids[i]
    return timeline.astype(int)

# %% ../notebooks/podverse/03_speaker_diarization.ipynb 22
def load_ids(mp3_path):
    return load(str(mp3_path) + "_speaker_ids.npy")