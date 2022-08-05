"""
This is an implementation of 

> "Denoising Diffusion Probabilistic Models", https://arxiv.org/abs/2006.11239

"""
import torch
from torch import nn
from pytorch_lightning import loggers
from .training import BatchLoss

sqrt = torch.sqrt    
tnsr = torch.tensor
nan  = float("nan")
N0   = lambda size, device=None: torch.normal(0, 1, size=size, device=None)


def beta_schedule(timesteps):
    """
    Generates a linear beta schedule starting at index 1.
    :param t_steps: number of diffusion steps to schedule betas for.
    :return: returns an array of length t_steps + 1 with the first element set to 0.0
    """
    assert timesteps > 0
    # paramters from https://arxiv.org/abs/2006.11239
    timesteps_base = 1000.0
    beta_start = 1e-4
    beta_end = 0.02

    # inverse-linearly scale beta with number of timesteps to make overall variance constant.
    scale = timesteps_base / timesteps
    betas = torch.linspace(beta_start * scale, beta_end * scale, timesteps)
    return torch.tensor([0.0, *betas])


def loss_simple(eps_hat, eps, t):
    """
    Simplified loss function from the paper, Eq. 14.
    :param eps: ground truth noise
    :param eps_hat: predicted noise
    """
    # Norm is taken over last 2 dims: width, and height.
    return torch.mean( torch.norm(eps - eps_hat, dim=(-2,-1))**2 )


def loss_general(eps_hat, eps, t, *, beta, alpha, alpha_bar, sig):
    """The $L_{t-1}$ term of the loss function from the paper (Eq. 12)."""
    w = beta[t]**2 / ( 2*sig[t]**2 * alpha[t] * (1 - alpha_bar[t]) )
    # Norm is taken over last 2 dims: width, and height.
    return torch.mean( w * torch.norm(eps - eps_hat, dim=(-2, -1))**2 )


def normalize_pixels(img: torch.Tensor) -> torch.Tensor:
    """
    Scales image intensities from [0., 1.] into range [-1., 1.].
    :param img: image with intensities in [0., 1.]
    """
    return 2 * img - 1.



class DiffusionModel(nn.Module, BatchLoss):
    def __init__(self, *, eps_model, timesteps=200, device=None, beta_scale=1.):
        super().__init__()
        nn.Module.__init__(self)
        BatchLoss.__init__(self, device=device)
        self.model = eps_model
        self.T     = timesteps
        self.register_buffer('beta', beta_schedule(self.T) * beta_scale)
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        self.to(device)


    def parameters(self):
        return self.model.parameters()


    def state_dict(self):
        return {
            'state_dict': nn.Module.state_dict(self),
            'hparams'   : self.hparams
        }


    def restore_state(self, state_dict):
        if 'state_dict' not in state_dict: raise Error('"state_dict" missing in state')
        self.load_state_dict(state_dict['state_dict'])

        if 'hparams' not in state_dict: raise Error('"hparams" missing in state')
        self.T = state_dict['hparams']['timesteps']


    def q_t(self, t, x_0 , *, eps=None):
        r"""Forward model $q(x_t \mid x_0)$ from Eq. 4."""
        if eps is None: eps = N0(x0.size())
        t = t.view(-1, 1, 1, 1) # enable compatability with x0
        return sqrt(self.alpha_bar[t]) * x_0  +  sqrt(1 - self.alpha_bar[t]) * eps


    def __call__(self, b, *, training_step: int = None, training_epoch: int = None):
        device = self.device

        x0, labels = b # img, labels
        B , _, _, _ = x0.size() # batch_size, channels, height, width

        t   = torch.randint(1, self.T + 1, size=(B,), device=device)
        x_0 = normalize_pixels(x0)
        eps = N0(x_0.size()).to(device)
        x_t = self.q_t(t=t, x_0=x_0, eps=eps)
        eps_hat = self.model(x_t, t, labels)

        L = loss_simple(eps_hat, eps, t)

        print(training_epoch, training_step,  L.item(), end='\r')

        return L


    @torch.no_grad()
    def denoise(self, x, *, t0=1, t1=None, labels=None, z_noise=N0, sigma=None, return_sequence=False, ts=None):
        """Algorithm 2 from the paper."""

        # Shorter notation to keep it simple.
        dev = self.device
        a, a_, sig, ls = self.alpha, self.alpha_bar, sigma, labels
        if sig is None: sig = sqrt(self.beta)
        if t1  is None: t1 = self.T
        if ls  is not None: ls = ls.to(dev)
        if ts is None:
            ts = torch.arange(t1, t0+1, step=-1, device=dev)

        # The sequence we're gonna attach the samples to
        xs      = [x.to(dev)]
        z_noise_ = lambda: z_noise(x.size()).to(dev)

        for t in ts:
            eps_model = self.model
            xt      = xs[-1]
            z       = z_noise_() if t >= 1 else 0.0
            eps_hat = eps_model(xt, t.unsqueeze(0), ls)
            mu      = 1/sqrt(a[t]) * (xt  - (1 - a[t])/sqrt(1 - a_[t]) * eps_hat) 
            x       = mu + sig[t]*z
            xs.append(x)

        if return_sequence: return torch.stack(xs)
        else: return xs[0]


    @torch.no_grad()
    def run_inference(self, *, t_start=0, x_start, labels, eps_fn, sigma=None, return_sequence=False):
        device = self.beta.get_device()
        eps = lambda: eps_fn().to(device)

        xs = [x_start.to(device)]
        l = labels.to(device)
        for t in reversed(range(t_start, self.T)):
            prefactor = sigma if sigma is not None else torch.sqrt(self.beta[t])
            z   = prefactor * eps() if t > 0 else 0.0
            eps_hat = self.model(xs[-1], torch.tensor([t], device=device), l)
            x   = 1 / torch.sqrt(self.alpha[t]) * (xs[-1] - (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t]) * eps_hat) + z
            if return_sequence:
                xs.append(x)
            else:
                xs[0] = x
        if return_sequence:
            return xs, l
        else:
            return xs[0], l