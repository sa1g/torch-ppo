import random
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Callable
import torch
import numpy as np
import gym.spaces as spaces
# pylint: disable=no-member

TensorDict = Dict[Union[str, int], torch.Tensor]

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

def update_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer: Pytorch optimizer
    :param learning_rate: New learning rate value
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")

def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        if type(observation_space.n) in [tuple, list, np.ndarray]:
            return tuple(observation_space.n)
        else:
            return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
        """
        Retrieve PyTorch device.
        It checks that the requested device is available first.
        For now, it supports only cpu and cuda.
        By default, it tries to use the gpu.

        Args:
            device: One for 'auto', 'cuda', 'cpu'
        
        Return
            Supported Pytorch device
        """
        # Cuda by default
        if device == "auto":
            device = "cuda"
        # Force conversion to torch.device
        device = torch.device(device)

        # Cuda not available
        if device.type == torch.device("cuda").type and not torch.cuda.is_available():
            return torch.device("cpu")

        return device

def obs_as_tensor(
    obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], device: torch.device
) -> Union[torch.Tensor, TensorDict]:
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: torch.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")