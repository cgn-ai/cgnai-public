from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union
import torch
from torch import nn as nn;
from torch.utils.data import DataLoader;
from torch.optim.optimizer import Optimizer
from pytorch_lightning import loggers
import inspect
from .hparams import HyperParameterSerializer
from .device_mixin import DeviceMixin
from pytorch_lightning import loggers




class BatchLoss(DeviceMixin, HyperParameterSerializer):
    """
    Main component of a training loop, a Callable that computes 
    the batch loss in the training loop.
    """
    def __init__(self, device=None):
        print(f"batchloss device:{device}")
        super().__init__(device=device)
        self.collect_hyperparameters()


    def __call__(self, b, *, training_step: int = None, training_epoch: int = None) -> torch.Tensor:
        raise NotImplementedError


    def parameters(self):
        raise NotImplementedError


    def state_dict(self):
        raise NotImplementedError


    def restore_state(self, state_dict):
        raise NotImplementedError


class StdBatchLoss(BatchLoss):
    def __init__(self, model, loss, device=None):
        super().__init__()
        self.model  = model.to(device) # No worries `decive=None` doesn't do harm
        self.loss   = loss
        self.device = device
        

    def __call__(self, 
                 b, 
                 *, 
                 training_step: int = None, 
                 training_epoch: int = None) -> torch.Tensor:
        x, y = b
        if not isinstance(x, tuple): x=(x,)
        if not isinstance(y, tuple): y=(y,)

        y_hat = self.model(*x)
        ell   = self.loss(y_hat, *y)
        return ell


    def parameters(self):
        return self.model.parameters()


    def state_dict(self):
        return self.model.state_dict()


    def restore_state(self, state_dict):
        self.model.load_state_dict(state_dict)



class TrainingLoop():
    """Training loop abstraction..."""
    def __init__(self,
                 batch_loss,
                 data_loader,
                 opt       = torch.optim.Adam,
                 logger    = None,
                 callbacks = {}): # Dict[event, List[Fn]]

        super().__init__()
        self.batch_loss  = batch_loss
        self.data_loader = data_loader
        self.logger      = logger
        self.callbacks   = callbacks

        self._step  = 0
        self._L     = [] # Training  losses

        #!
        #!  Construct optimizer.
        #!
        if not isinstance(opt, Tuple): opt = (opt,)
        opt, *opt_ = opt
        opt_args   = {} 
        for x in opt_:
            if isinstance(x, torch.nn.Module): opt_args["params"] = x.parameters()
            if isinstance(x, list): opt_args["params"] = x
            if isinstance(x, dict): opt_args.update(x)
        #! if optimizer didn't get explicit params defer to `batch_loss`
        if "params" not in opt_args: 
            opt_args["params"] = self.batch_loss.parameters()
        self.optimizer = opt(**opt_args)


    def _emit(self, event):
        if event in self.callbacks:
            for cb in self.callbacks[event]:
                cb()


    def on_run_start(self):
        if self.logger is not None: self.logger.save()


    def on_batch_end(self, *a, **kw):
        if self._step % 50 == 0:
            e, b, dl, L = self._epoch, self._batch_step, self.data_loader, self._L
            print(f"e:{e} b:{e}/{len(dl)} ell:{L[-1][-1]:0.5f}", end="\r")


    def on_epoch_start(self):
        if self.logger is not None: self.logger.save()


    def on_epoch_end(self):
        if self.logger is not None: self.logger.save()


    def on_run_end(self):
        if self.logger is not None: 
            path = Path(self.logger.save_dir) / f'version_{self.logger.version}'
            self.store_state(path / 'state.tar')


    def run(self, epochs, max_steps=float("inf")):
        """
        Runs the training loop for a given number of epochs,
        but not more than a maximum number of steps. 
        """
        self.on_run_start()

        for epoch in range(epochs):
            self._epoch = epoch + 1
            self._L.append([])
            self.on_epoch_start()

            for i, b in enumerate(self.data_loader):
                # Keeping track of the total 
                # and current training amount
                self._step += 1
                self._batch_step = i + 1
                if self._step > max_steps: return self.L

                # The actual training
                l = self.batch_loss(b, training_step=self._step, training_epoch=self._epoch)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                self._L[-1].append(l.item())
                self.on_batch_end()

            self._emit('epoch_end')
            self.on_epoch_end() # TODO move this to callback.
            
        self._emit('training_end')
        self.on_run_end()  # TODO move this to callback.
        return self.L


    def store_state(self, filepath):
        checkpoint = {
            'batch_loss_state_dict': self.batch_loss.state_dict(),
            'optimizer_state_dict':  self.optimizer.state_dict(),
            'training_loop_state_dict': {
                'L':    self._L,
                'step': self._step,
            }
        }
        torch.save(checkpoint, filepath)
        print(f"State stored in {filepath}")


    def restore_from_checkpoint(self, state_filename):
        state_dict = torch.load(state_filename)

        # 1) restore batch loss state
        if 'batch_loss_state_dict' not in state_dict:
            raise Error('"batch_loss_state_dict" missing in state')
        self.batch_loss.restore_state(state_dict['batch_loss_state_dict'])

        # 2) restore optimizer state
        if 'optimizer_state_dict' not in state_dict:
            raise Error('"optimizer_state_dict" missing in state')
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        # 3) training loop state
        if 'training_loop_state_dict' not in state_dict:
            raise Error('"training_loop_state_dict" missing in state')
        self._L = state_dict['training_loop_state_dict']['L']
        self._step = state_dict['training_loop_state_dict']['step']


    @property
    def L(self):
        return self._L


    @property
    def L_flatten(self):
        return torch.cat([torch.tensor(l) for l in self._L])