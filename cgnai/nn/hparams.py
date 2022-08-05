import inspect
import types

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Literal


SUPPORTED_HPARAM_TYPES = (
    int,
    str,
    float,
    dict
)


def parse_class_init_keys(
    cls: Union[Type["pl.LightningModule"], Type["pl.LightningDataModule"]]
) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse key words for standard ``self``, ``*args`` and ``**kwargs``.
    Examples:
        >>> class Model():
        ...     def __init__(self, hparams, *my_args, anykw=42, **my_kwargs):
        ...         pass
        >>> parse_class_init_keys(Model)
        ('self', 'my_args', 'my_kwargs')
    """
    init_parameters = inspect.signature(cls.__init__).parameters
    # docs claims the params are always ordered
    # https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
    init_params = list(init_parameters.values())
    # self is always first
    n_self = init_params[0].name

    def _get_first_if_any(
        params: List[inspect.Parameter],
        param_type: Literal[inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD],
    ) -> Optional[str]:
        for p in params:
            if p.kind == param_type:
                return p.name
        return None

    n_args = _get_first_if_any(init_params, inspect.Parameter.VAR_POSITIONAL)
    n_kwargs = _get_first_if_any(init_params, inspect.Parameter.VAR_KEYWORD)

    return n_self, n_args, n_kwargs


def get_init_args(frame: types.FrameType) -> Dict[str, Any]:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if "__class__" not in local_vars:
        return {}
    cls = local_vars["__class__"]
    init_parameters = inspect.signature(cls.__init__).parameters
    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    filtered_vars = [n for n in (self_var, args_var, kwargs_var) if n]
    exclude_argnames = (*filtered_vars, "__class__", "frame", "frame_args")
    # only collect variables that appear in the signature
    local_args = {k: local_vars[k] for k in init_parameters.keys()}
    # kwargs_var might be None => raised an error by mypy
    if kwargs_var:
        local_args.update(local_args.get(kwargs_var, {}))
    local_args = {k: v for k, v in local_args.items() if k not in exclude_argnames}

    # filter out non-supported types
    return {k: v for k, v in local_args.items() if type(v) in SUPPORTED_HPARAM_TYPES}


def collect_init_args(
    frame: types.FrameType,
    path_args: List[Dict[str, Any]],
    inside: bool = False,
    classes: Tuple[Type, ...] = (),
) -> List[Dict[str, Any]]:
    _, _, _, local_vars = inspect.getargvalues(frame)
    # frame.f_back must be of a type types.FrameType for get_init_args/collect_init_args due to mypy
    if not isinstance(frame.f_back, types.FrameType):
        return path_args

    if "__class__" in local_vars and (not classes or issubclass(local_vars["__class__"], classes)):
        local_args = get_init_args(frame)
        # recursive update
        path_args.append(local_args)
        return collect_init_args(frame.f_back, path_args, inside=True, classes=classes)
    if not inside:
        return collect_init_args(frame.f_back, path_args, inside, classes=classes)
    return path_args


def save_hyperparameters(
    obj: Any, frame: Optional[types.FrameType] = None
) -> None:
    if is_dataclass(obj):
        init_args = {f.name: getattr(obj, f.name) for f in fields(obj)}
    else:
        init_args = {}
        for local_args in collect_init_args(frame, [], classes=(HyperParameterSerializer,)):
            init_args.update(local_args)
    return init_args


class HyperParameterSerializer(object):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._log_hyperparams = False

    def collect_hyperparameters(self, frame: Optional[types.FrameType] = None) -> None:
        self._log_hyperparams = True
        if not frame:
            current_frame = inspect.currentframe()
            # inspect.currentframe() return type is Optional[types.FrameType]: current_frame.f_back called only if available
            if current_frame:
                frame = current_frame.f_back
        if not isinstance(frame, types.FrameType):
            raise AttributeError("There is no `frame` available while being required.")
        self.hparams = save_hyperparameters(self, frame=frame)
        print(f'hparams:\n{self.hparams}')