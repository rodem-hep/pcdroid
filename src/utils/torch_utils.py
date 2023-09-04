"""Mix of utility functions specifically for pytorch."""
import os
from functools import partial
from typing import Union

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim.lr_scheduler as schd

from src.utils.schedulers import (
    CyclicWithWarmup,
    LinearWarmupRootDecay,
    WarmupToConstant,
)

# An onnx save argument which is for the pass with mask function (makes it slower)
ONNX_SAFE = False


def append_dims(x: T.Tensor, target_dims: int, add_to_front: bool = False) -> T.Tensor:
    """Appends dimensions of size 1 to the end or front of a tensor until it
    has target_dims dimensions.

    Parameters
    ----------
    x : T.Tensor
        The input tensor to be reshaped.
    target_dims : int
        The target number of dimensions for the output tensor.
    add_to_front : bool, optional
        If True, dimensions are added to the front of the tensor.
        If False, dimensions are added to the end of the tensor.
        Defaults to False.

    Returns
    -------
    T.Tensor
        The reshaped tensor with target_dims dimensions.

    Raises
    ------
    ValueError
        If the input tensor already has more dimensions than target_dims.

    Examples
    --------
    >>> x = T.tensor([1, 2, 3])
    >>> x.shape
    torch.Size([3])

    >>> append_dims(x, 3)
    tensor([[[1]], [[2]], [[3]]])
    >>> append_dims(x, 3).shape
    torch.Size([3, 1, 1])

    >>> append_dims(x, 3, add_to_front=True)
    tensor([[[[1, 2, 3]]]])
    >>> append_dims(x, 3, add_to_front=True).shape
    torch.Size([1, 1, 3])
    """
    dim_diff = target_dims - x.dim()
    if dim_diff < 0:
        raise ValueError(f"x has more dims ({x.ndim}) than target ({target_dims})")
    if add_to_front:
        return x[(None,) * dim_diff + (...,)]  # x.view(*dim_diff * (1,), *x.shape)
    return x[(...,) + (None,) * dim_diff]  # x.view(*x.shape, *dim_diff * (1,))


class GradsOff:
    """Context manager for passing through a model without it tracking
    gradients."""

    def __init__(self, model) -> None:
        self.model = model

    def __enter__(self) -> None:
        self.model.requires_grad_(False)

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.model.requires_grad_(True)


def get_act(name: str) -> nn.Module:
    """Return a pytorch activation function given a name."""
    if isinstance(name, partial):
        return name()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "lrlu":
        return nn.LeakyReLU(0.1)
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "selu":
        return nn.SELU()
    if name == "softmax":
        return nn.Softmax()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softmax":
        return nn.Softmax()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError("No activation function with name: ", name)


def get_nrm(name: str, outp_dim: int) -> nn.Module:
    """Return a 1D pytorch normalisation layer given a name and a output size
    Returns None object if name is none."""
    if name == "batch":
        return nn.BatchNorm1d(outp_dim)
    if name == "layer":
        return nn.LayerNorm(outp_dim)
    if name == "none":
        return None
    else:
        raise ValueError("No normalistation with name: ", name)


def get_loss_fn(name: Union[partial, str], **kwargs) -> nn.Module:
    """Return a pytorch loss function given a name."""

    # Supports using partial methods instad of having to do support each string
    if isinstance(name, partial):
        return name()
    if name == "none":
        return None
    if name == "crossentropy":
        return nn.CrossEntropyLoss(reduction="none")
    if name == "huber":
        return nn.HuberLoss(reduction="none")
    if name == "mse":
        return nn.MSELoss(reduction="none")
    if name == "mae":
        return nn.L1Loss(reduction="none")
    else:
        raise ValueError(f"No standard loss function with name: {name}")


def get_sched(
    sched_dict,
    opt,
    steps_per_epoch: int = 0,
    max_lr: float = None,
    max_epochs: float = None,
) -> schd._LRScheduler:
    """Return a pytorch learning rate schedular given a dict containing a name
    and other kwargs.

    I still prefer this method as opposed to the hydra implementation as
    it allows us to specify the cyclical scheduler periods as a function of epochs
    rather than steps.

    args:
        sched_dict: A dictionary of kwargs used to select and configure the schedular
        opt: The optimiser to apply the learning rate to
        steps_per_epoch: The number of minibatches in a training single epoch
    kwargs: (only for OneCyle learning!)
        max_lr: The maximum learning rate for the one shot
        max_epochs: The maximum number of epochs to train for
    """

    # Pop off the name and learning rate for the optimiser
    dict_copy = sched_dict.copy()
    name = dict_copy.pop("name")

    # Get the max_lr from the optimiser if not specified
    max_lr = max_lr or opt.defaults["lr"]

    # Exit if the name indicates no scheduler
    if name in ["", "none", "None"]:
        return None

    # If the steps per epoch is 0, try and get it from the sched_dict
    if steps_per_epoch == 0:
        try:
            steps_per_epoch = dict_copy.pop("steps_per_epoch")
        except KeyError:
            raise ValueError(
                "steps_per_epoch was not passed to get_sched and was ",
                "not in the scheduler dictionary!",
            )

    # Pop off the number of epochs per cycle (needed as arg)
    if "epochs_per_cycle" in dict_copy:
        epochs_per_cycle = dict_copy.pop("epochs_per_cycle")
    else:
        epochs_per_cycle = 1

    # Use the same div_factor for cyclic with warmup
    if name == "cyclicwithwarmup":
        if "div_factor" not in dict_copy:
            dict_copy["div_factor"] = 1e4

    if name == "cosann":
        return schd.CosineAnnealingLR(
            opt, steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    elif name == "cosannwr":
        return schd.CosineAnnealingWarmRestarts(
            opt, steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    elif name == "onecycle":
        return schd.OneCycleLR(
            opt, max_lr, total_steps=steps_per_epoch * max_epochs, **dict_copy
        )
    elif name == "cyclicwithwarmup":
        return CyclicWithWarmup(
            opt, max_lr, total_steps=steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    elif name == "linearwarmuprootdecay":
        return LinearWarmupRootDecay(opt, **dict_copy)
    elif name == "warmup":
        return WarmupToConstant(opt, **dict_copy)
    elif name == "lr_sheduler.ExponentialLR":
        return schd.ExponentialLR(opt, **dict_copy)
    elif name == "lr_scheduler.ConstantLR":
        return schd.ConstantLR(opt, **dict_copy)
    else:
        raise ValueError(f"No scheduler with name: {name}")


def sel_device(dev: Union[str, T.device]) -> T.device:
    """Returns a pytorch device given a string (or a device)

    - giving cuda or gpu will run a hardware check first
    """
    # Not from config, but when device is specified already
    if isinstance(dev, T.device):
        return dev

    # Tries to get gpu if available
    if dev in ["cuda", "gpu"]:
        print("Trying to select cuda based on available hardware")
        dev = "cuda" if T.cuda.is_available() else "cpu"

    # Tries to get specific gpu
    elif "cuda" in dev:
        print(f"Trying to select {dev} based on available hardware")
        dev = dev if T.cuda.is_available() else "cpu"

    print(f"Running on hardware: {dev}")
    return T.device(dev)


def move_dev(
    tensor: Union[T.Tensor, tuple, list, dict], dev: Union[str, T.device]
) -> Union[T.Tensor, tuple, list, dict]:
    """Returns a copy of a tensor on the targetted device. This function calls
    pytorch's .to() but allows for values to be a.

    - list of tensors
    - tuple of tensors
    - dict of tensors
    """

    # Select the pytorch device object if dev was a string
    if isinstance(dev, str):
        dev = sel_device(dev)

    if isinstance(tensor, tuple):
        return tuple(t.to(dev) for t in tensor)
    elif isinstance(tensor, list):
        return [t.to(dev) for t in tensor]
    elif isinstance(tensor, dict):
        return {t: tensor[t].to(dev) for t in tensor}
    else:
        return tensor.to(dev)


def to_np(inpt: Union[T.Tensor, tuple]) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a pytorch
    tensor to numpy array.

    - Includes gradient deletion, and device migration
    """
    if inpt is None:
        return None
    if isinstance(inpt, (tuple, list)):
        return type(inpt)(to_np(x) for x in inpt)
    if inpt.dtype == T.bfloat16:  # Numpy conversions don't support bfloat16s
        inpt = inpt.half()
    return inpt.detach().cpu().numpy()


def print_gpu_info(dev=0):
    """Prints current gpu usage."""
    total = T.cuda.get_device_properties(dev).total_memory / 1024**3
    reser = T.cuda.memory_reserved(dev) / 1024**3
    alloc = T.cuda.memory_allocated(dev) / 1024**3
    print(f"\nTotal = {total:.2f}\nReser = {reser:.2f}\nAlloc = {alloc:.2f}")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a pytorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_grad_norm(model: nn.Module, norm_type: float = 2.0):
    """Return the norm of the gradients of a given model."""
    return to_np(
        T.norm(
            T.stack([T.norm(p.grad.detach(), norm_type) for p in model.parameters()]),
            norm_type,
        )
    )


def get_max_cpu_suggest():
    """try to compute a suggested max number of worker based on system's
    resource."""
    max_num_worker_suggest = None
    if hasattr(os, "sched_getaffinity"):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        max_num_worker_suggest = os.cpu_count()
    return max_num_worker_suggest


def log_squash(data: T.Tensor) -> T.Tensor:
    """Apply a log squashing function for distributions with high tails."""
    return T.sign(data) * T.log(T.abs(data) + 1)


def torch_undo_log_squash(data: np.ndarray) -> np.ndarray:
    """Undo the log squash function above."""
    return T.sign(data) * (T.exp(T.abs(data)) - 1)


@T.no_grad()
def ema_param_sync(source: nn.Module, target: nn.Module, ema_decay: float) -> None:
    """Synchronize the parameters of two modules using exponential moving
    average (EMA).

    Parameters
    ----------
    source : nn.Module
        The source module whose parameters are used to update the target module.
    target : nn.Module
        The target module whose parameters are updated.
    ema_decay : float
        The decay rate for the EMA update.
    """
    for s_params, t_params in zip(source.parameters(), target.parameters()):
        t_params.data.copy_(
            ema_decay * t_params.data + (1.0 - ema_decay) * s_params.data
        )
