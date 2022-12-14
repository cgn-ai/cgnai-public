# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/podverse/01_audio_core.ipynb.

# %% auto 0
__all__ = ['ToMel', 'cut_up', 'to_samples', 'to_ms']

# %% ../notebooks/podverse/01_audio_core.ipynb 3
import warnings
warnings.filterwarnings("ignore")
import torch
from ..utils import cgnai_home, sliding_window_ind
from torchvision.transforms import Compose
from torchaudio.transforms import Resample, MelSpectrogram

# %% ../notebooks/podverse/01_audio_core.ipynb 4
def ToMel(sr, wav_cut_spec:"(width, displacement)", n_mels:"features", ):
    """Maps a wav signal to its melspectrum."""
    w, d = wav_cut_spec
    
    components = []
        
    components.append(MelSpectrogram(
        sample_rate = sr, 
        n_fft       = w, 
        hop_length  = d, 
        pad         = 0, 
        n_mels      = n_mels, 
        normalized  = False))

    components.append(torchaudio.transforms.AmplitudeToDB(
        stype  = 'power', 
        top_db =  80))

    to_mel = Compose(components)
    
    return to_mel

# %% ../notebooks/podverse/01_audio_core.ipynb 5
def cut_up(x, cut_spec:"(width, displacement)"):
    """
    Cuts up an array along its last(!!!) dimension
    according to the cut spec - a tuple of 
    width and displacement.
    
    Note: The name sucks because it 
    really just is sliding windows, 
    and not "cuts" ...äaaanyway.
    """
    # move last dim to beginning
    # so we can access the cutting 
    # dimension easily, x[I]
    dims = tuple(range(x.dim()))
    perm = dims[-1:] + dims[:-1]
    x = torch.permute(x, perm)
    
    w, d = cut_spec
    T = x.size(0)

    I = sliding_window_ind(T, w, step=d)
    I = torch.tensor(I)
    c = x[I]

    # move first dim to end
    # - get back the original shape + cutdim
    dims = tuple(range(c.dim()))
    perm = (dims[0], *dims[2:], dims[1])
    c = torch.permute(c, perm)
    return c

# %% ../notebooks/podverse/01_audio_core.ipynb 6
def to_samples(i, wav_cut_spec:"(width, displacement)", mel_cut_spec:"(width, displacement)"):
    """Mel-cut index to sample index."""
    w , d  = wav_cut_spec
    w_, d_ = mel_cut_spec
    return i*d_*d, (i*d_ + w_)*d + w
            
def to_ms(s, sr):
    """From samples at a certain rate (Hz) to ms"""
    return s/sr*1000
