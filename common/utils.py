"""
Borrow from stable-baselines3
Due to dependencies incompability, we cherry-pick codes here
"""
import os, random, re
from datetime import datetime
import warnings
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from gymnasium import spaces

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).
    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).
    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        warnings.warn("Treating image space as channels-last, while second dimension was smallest of the three.")
    return smallest_dimension == 0


def is_image_space(
    observation_space: spaces.Space,
    check_channels: bool = False,
    normalized_image: bool = False,
) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.
    Valid images: RGB, RGBD, GrayScale with values in [0, 255]
    :param observation_space:
    :param check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    :return:
    """
    check_dtype = check_bounds = not normalized_image
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if check_dtype and observation_space.dtype != np.uint8:
            return False

        # Check the value range
        incorrect_bounds = np.any(observation_space.low != 0) or np.any(observation_space.high != 255)
        if check_bounds and incorrect_bounds:
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # GrayScale, RGB, RGBD
        return n_channels in [1, 3, 4]
    return False



def preprocess_obs(
    obs: torch.Tensor,
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.
    :param obs: Observation
    :param observation_space:
    :param normalize_images: Whether to normalize images or not
        (True by default)
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        if normalize_images and is_image_space(observation_space):
            return obs.float() / 255.0
        return obs.float()

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return torch.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(torch.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()

    elif isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        assert isinstance(obs, Dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key], normalize_images=normalize_images)
        return preprocessed_obs

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")


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


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.
    Used by the ``FlattenExtractor`` to compute the input shape.
    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)


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
    elif isinstance(action_space, spaces.Dict):
        return get_action_dim(action_space["motor_action"])
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def check_for_nested_spaces(obs_space: spaces.Space):
    """
    Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    If so, raise an Exception informing that there is no support for this.
    :param obs_space: an observation space
    :return:
    """
    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = obs_space.spaces.values() if isinstance(obs_space, spaces.Dict) else obs_space.spaces
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)."
                )


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.
    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
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


def get_timestr() -> str:
    current_datetime = datetime.now()
    return current_datetime.strftime("%m-%d-%H-%M-%S")


def get_spatial_emb_indices(loc: np.ndarray,
                            full_img_size=(4, 84, 84), 
                            img_size=(4, 21, 21), 
                            patch_size=(7, 7)) -> np.ndarray:
    # loc (2,)
    _, H, W = full_img_size
    _, h, w = img_size
    p1, p2 = patch_size

    st_x = loc[0] // p1
    st_y = loc[1] // p2

    ed_x = (loc[0] + h) // p1
    ed_y = (loc[1] + w) // p2

    ix, iy = np.meshgrid(np.arange(st_x, ed_x, dtype=np.int64),
                            np.arange(st_y, ed_y, dtype=np.int64), indexing="ij")

    # print (ix, iy)
    indicies = (ix * H // p1 + iy).reshape(-1)

    return indicies

def get_spatial_emb_mask(loc, 
                         mask,
                         full_img_size=(4, 84, 84), 
                         img_size=(4, 21, 21), 
                         patch_size=(7, 7), 
                         latent_dim=144) -> np.ndarray:
    B, T, _ = loc.size()
    # return torch.randn_like()
    loc = loc.reshape(-1, 2)
    _, H, W = full_img_size
    _, h, w = img_size
    p1, p2 = patch_size
    num_tokens = h*w//p1//p2
    # print ("num_tokens", num_tokens)

    st_x = loc[..., 0] // p1
    st_y = loc[..., 1] // p2

    ed_x = (loc[..., 0] + h) // p1
    ed_y = (loc[..., 1] + w) // p2

    # mask = np.zeros(((32*6, H//p1, W//p2, latent_dim)), dtype=np.bool_)
    # mask = torch.zeros((32*6, H//p1, W//p2, latent_dim), dtype=torch.bool)
    mask[:] = False
    for i in range(B*T):
        # print (self.spatial_emb[0, st_x[i]:ed_x[i], st_y[i]:ed_y[i]].size())
        mask[i, st_x[i]:ed_x[i], st_y[i]:ed_y[i]] = True
    return mask[:B*T]

def weight_init_drq(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)
        
class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
    

def schedule_drq(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

def get_sugarl_reward_scale_robosuite(task_name) -> float:
    if task_name == "Lift":
        sugarl_reward_scale = 150/500
    elif task_name == "ToolHang":
        sugarl_reward_scale = 100/500
    else:
        sugarl_reward_scale = 100/500
    return sugarl_reward_scale


def get_sugarl_reward_scale_dmc(domain_name, task_name) -> float:
    if domain_name == "ball_in_cup" and task_name == "catch":
        sugarl_reward_scale = 320/500
    elif domain_name == "cartpole" and task_name == "swingup":
        sugarl_reward_scale = 380/500
    elif domain_name == "cheetah" and task_name == "run":
        sugarl_reward_scale = 245/500
    elif domain_name == "dog" and task_name == "fetch":
        sugarl_reward_scale = 4.5/500
    elif domain_name == "finger" and task_name == "spin":
        sugarl_reward_scale = 290/500
    elif domain_name == "fish" and task_name == "swim":
        sugarl_reward_scale = 64/500
    elif domain_name == "reacher" and task_name == "easy":
        sugarl_reward_scale = 200/500
    elif domain_name == "walker" and task_name == "walk":
        sugarl_reward_scale = 290/500
    else:
        return 1.
    
    return sugarl_reward_scale

def get_sugarl_reward_scale_atari(game) -> float:
    base_scale = 4.0
    sugarl_reward_scale = 1/200
    if game in ["alien", "assault", "asterix", "battle_zone", "seaquest", "qbert", "private_eye", "road_runner"]:
        sugarl_reward_scale = 1/100
    elif game in ["kangaroo", "krull", "chopper_command", "demon_attack"]:
        sugarl_reward_scale = 1/200
    elif game in ["up_n_down", "frostbite", "ms_pacman", "amidar", "gopher", "boxing"]:
        sugarl_reward_scale = 1/50
    elif game in ["hero", "jamesbond", "kung_fu_master"]:
        sugarl_reward_scale = 1/25
    elif game in ["crazy_climber"]:
        sugarl_reward_scale = 1/20
    elif game in ["freeway"]:
        sugarl_reward_scale = 1/1600
    elif game in ["pong"]:
        sugarl_reward_scale = 1/800
    elif game in ["bank_heist"]:
        sugarl_reward_scale = 1/250
    elif game in ["breakout"]:
        sugarl_reward_scale = 1/35
    sugarl_reward_scale = sugarl_reward_scale * base_scale
    return sugarl_reward_scale
