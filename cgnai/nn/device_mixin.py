from decorator import decorator
import inspect
import wrapt
from torch import Tensor

@wrapt.decorator
def patch_call(f, instance, args, kwargs):
    """
    Puts the batch on device if set, and 
    ensures that the batch is of the form `(x,y)`,
    with `y` potentially being `None`.
    """
    b, *args_ = args

    def to_dev(T): return T.to(instance.device)

    if isinstance(b, Tensor):
        x = to_dev(b)
        b = (x, None)

    elif isinstance(b, tuple) and len(b) == 1:
        x, = b
        if isinstance(x, tuple): tuple(map(to_dev, x))
        else: x = to_dev(x)    
        b = (x, None)

    else:            
        x, y = b
        if isinstance(x, tuple): tuple(map(to_dev, x))
        else: x = to_dev(x)
        if isinstance(y, tuple): tuple(map(to_dev, y))
        else: y = to_dev(y)
        b = (x, y)
    
    return f(b, *args_, **kwargs)
    

class DeviceMixin(object):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        if device is not None: self._patch_call()


    def _patch_call(self):
        ATTR = "_DeviceMixin__call"
        if not hasattr(self.__class__, ATTR):
            setattr(self.__class__, ATTR, self.__class__.__call__)
            self.__class__.__call__ = patch_call(self.__class__.__call__)


def device_deco(C):
    ATTR = "_DeviceMixin__call"
    if not hasattr(C, ATTR):
        setattr(C, ATTR, C.__call__)
        C.__call__ = patch_call(C.__call__)

    return C
