import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import OneHotCategoricalStraightThrough

import os
from typing import List, Dict, Tuple, Callable

def set_seed(config):
    """Seed the program."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)

def init_device(config):
    if config.device.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        if config.device.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    torch.set_num_threads(config.device.torch_threads)
    return device

def n2t(input: np.ndarray, **kwargs) -> torch.Tensor:
    assert isinstance(input, np.ndarray)
    input = convert(input)
    return torch.tensor(input, **kwargs)

def merge_dict_list(
        dict_list: List[Dict[str, np.ndarray | torch.Tensor]],
        axis: int = 0,
        is_tensor: bool = False,
        keys: List[str] | None = None,
    ) -> Dict[str, np.ndarray]:
    if keys is None:
        keys = dict_list[0].keys()
    if is_tensor:
        return {
            k: torch.stack([d[k] for d in dict_list], dim=axis)
            for k in keys if not k.startswith("log_")
        }
    else:
        return {
            k: np.stack([convert(d[k]) for d in dict_list], axis=axis)
            for k in keys if not k.startswith("log_")
        }

def scan(
    fn: Callable,
    inputs: Tuple[torch.Tensor, ...] | None = None,
    init_state: Dict[str, torch.Tensor] | None = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Recursively applies a function to a sequence of inputs and an initial state, accumulating and returning the list of intermediate states.
    """
    inputs = [torch.unbind(input, dim=0) for input in inputs]
    inputs = list(zip(*inputs))
    states = []

    state = init_state
    for input in inputs:
        state = fn(*input, state)
        states.append(state)
    return states

class RequiresGrad:
    def __init__(self, model, initial=False):
        self._model = model
        self._model.requires_grad_(requires_grad=initial)

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, exc_type, exc_value, traceback):
        self._model.requires_grad_(requires_grad=False)

class Optimizer:
    def __init__(self, name, parameters, lr, eps, use_max_grad_norm, max_grad_norm):
        self._name = name
        self._parameters = list(parameters)
        self._use_max_grad_norm = use_max_grad_norm
        self._max_grad_norm = max_grad_norm
        self._optim = torch.optim.Adam(
            self._parameters,
            lr=lr,
            eps=eps,
        )

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        self._optim.load_state_dict(state_dict)

    def __call__(self, loss) -> Dict:
        self._optim.zero_grad()
        loss.backward()
        if self._use_max_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(self._parameters, self._max_grad_norm)
        else:
            grad_norm = get_grad_norm(self._parameters)
        self._optim.step()
        metrics = {
            f"{self._name}_loss": loss.item(),
            f"{self._name}_grad_norm": grad_norm.item(),
        }
        return metrics

def get_grad_norm(parameters):
    grads = [p.grad for p in parameters if p.grad is not None]
    norms = [torch.linalg.vector_norm(grad.detach(), ord=2) for grad in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    return total_norm

def weight_init(m: nn.Module, method: str = "truncated_normal", scale: float = 1.0) -> None:
    if isinstance(m, nn.Linear):
        in_dim = m.in_features
        out_dim = m.out_features
        avg = (in_dim + out_dim) / 2
        if method == "uniform":
            limit = scale * np.sqrt(3 / avg)
            nn.init.uniform_(m.weight, a=-limit, b=limit)
        elif method == "truncated_normal":
            nn.init.trunc_normal_(m.weight, a=-2, b=2)
            m.weight.data *= 1.1368 * np.sqrt(1 / avg)
        else:
            raise NotImplementedError

        if hasattr(m, "bias") and hasattr(m.bias, "data"):
            nn.init.zeros_(m.bias)

        # scale the weight
        m.weight.data *= scale
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_dim = space * m.in_channels
        out_dim = space * m.out_channels
        avg = (in_dim + out_dim) / 2
        if method == "uniform":
            limit = scale * np.sqrt(3 / avg)
            nn.init.uniform_(m.weight, a=-limit, b=limit)
        elif method == "truncated_normal":
            nn.init.trunc_normal_(m.weight, a=-2, b=2)
            m.weight.data *= 1.1368 * np.sqrt(1 / avg)
        else:
            raise NotImplementedError

        if hasattr(m, "bias") and hasattr(m.bias, "data"):
            nn.init.zeros_(m.bias)

        # scale the weight
        m.weight.data *= scale
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        raise NotImplementedError
    elif isinstance(m, nn.Parameter):
        raise NotImplementedError

def build_returns(
    rewards: torch.Tensor,
    value_preds: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    returns = value_preds.clone()
    gae = 0
    for step in reversed(range(rewards.shape[0] - 1)):
        delta = (
            rewards[step + 1]
            + gamma * value_preds[step + 1] * (1 - terminated[step + 1])
            - value_preds[step]
        )
        gae = delta + gamma * gae_lambda * gae * (1 - terminated[step + 1]) * (1 - truncated[step + 1])
        returns[step] += gae
    return returns

def latent_to_input(latent: Dict[str, torch.Tensor]) -> torch.Tensor:
    input = torch.cat(
        [
            latent["deter"],
            latent["stoch"].flatten(-2, -1),
        ],
        dim=-1,
    )
    return input

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def get_task_name(config):
    if config.env == "smac":
        return config.env_args.map_name
    elif config.env == "smacv2":
        return config.env_args.map_name
    elif "pettingzoo" in config.env:
        return config.env_args.lib_name + "." + config.env_args.env_name
    elif config.env == "meltingpot":
        return config.env_args.substrate
    else:
        raise ValueError(f"Environment {config.env} not supported")

def make_env(config, index):
    if config.env == "smac":
        from envs.smac_wrapper import SMACWrapper
        seed = config.seed + index * 10
        env = SMACWrapper(
            **config.env_args,
            seed=seed,
        )
    elif config.env == "smacv2":
        from envs.smacv2_wrapper import SMACv2Wrapper
        seed = config.seed + index * 10
        env = SMACv2Wrapper(
            **config.env_args,
            seed=seed,
        )
    elif "pettingzoo" in config.env:
        from envs.pettingzoo_wrapper import PettingZooWrapper
        env = PettingZooWrapper(
            **config.env_args,
            seed=config.seed + index * 10,
        )
    elif config.env == "meltingpot":
        from envs.meltingpot_wrapper import MeltingPotWrapper
        env = MeltingPotWrapper(
            config.env_args,
            # seed=config.seed + index * 10, # TODO
        )
    else:
        raise ValueError(f"Environment {config.env} not supported")
    return env

def get_st_onehot(logits: torch.Tensor, unimix: float = 0.0):
    probs = torch.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.size(-1)
    probs = probs * (1 - unimix) + uniform * unimix
    return OneHotCategoricalStraightThrough(probs=probs)

class UniformSampler:
    def __init__(self):
        self.indices = {}
        self.keys = []

    def __len__(self):
        return len(self.keys)

    def __call__(self, key):
        assert key not in self.indices, (key, self.indices[key])
        self.indices[key] = len(self.keys)
        self.keys.append(key)

    def __delitem__(self, key):
        assert 2 <= len(self), len(self)
        index = self.indices.pop(key)
        last = self.keys.pop()
        if index != len(self.keys):
            self.keys[index] = last
            self.indices[last] = index

    def __contains__(self, key):
        return key in self.indices

    def sample(self):
      index = np.random.randint(len(self.keys))
      return self.keys[index]
