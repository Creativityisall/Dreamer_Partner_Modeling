import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensordict.tensordict import TensorDict

from typing import Dict
from collections import defaultdict

from utils.tools import Optimizer, RequiresGrad
from utils.networks import MLP, weight_init, Partner_Model
import elements
class Global_Encoder(nn.Module):
    _name = "global_encoder"
    def __init__(self, config, n_agents, device=torch.device("cpu")):
        super().__init__()
        self.config = config
        self.n_agents = n_agents
        self.device = device

        # encoder
        self.encoder = MLP(
            in_dim=(
                config.world_model.rssm.deterministic_dim * n_agents
                + config.world_model.rssm.stochastic_dim * config.world_model.rssm.classes * n_agents# 这里的参数表达的是这是输出的离散类型随机变量经过one-hot后的结果。
            ),                                                                             # 每个随机变量有stochastic_dim，每个维度可能是classes个整数中的一个
            hidden_dim=config.global_encoder.hidden_dim,
            hidden_layers=config.global_encoder.hidden_layers,
            out_dim=config.global_encoder.hidden_dim * n_agents,
            act=config.global_encoder.act,
            use_layernorm=config.global_encoder.use_layernorm,
            device=device,
        )

        # context manager for gradient control
        self._requires_grad = RequiresGrad(self)

        # optimizer
        self._optim = Optimizer(
            name=self._name,
            parameters=self.parameters(),
            lr=config.train.global_encoder.lr,
            eps=config.train.global_encoder.eps,
            use_max_grad_norm=config.train.global_encoder.use_max_grad_norm,
            max_grad_norm=config.train.global_encoder.max_grad_norm,
        )

    def forward(
            self,
            latent: Dict[str, torch.Tensor],# TODO:全局的h和s
            evaluation: bool = False,
        ) -> TensorDict:
        deter = latent["deter"].flatten(start_dim=1)# TODO:全局的h和s
        stoch = latent["stoch"].flatten(start_dim=1)
        x = torch.cat([deter, stoch], dim=-1)

        # 训练时进行编码
        if not evaluation:
            global_vector = self.encoder(x)
        else:
            raise NotImplementedError

        return global_vector

    def update(
            self,
            global_vectors: torch.Tensor,
            local_vectors: torch.Tensor,
    ):
        A = local_vectors.shape[1]
        expended_global_vectors = global_vectors.unsqueeze(1).expand(-1, A, -1)
        mse_loss_fn = nn.MSELoss(reduction='mean')
        global_partner_loss = mse_loss_fn(local_vectors.detach(), expended_global_vectors)
        self._optim(global_partner_loss)