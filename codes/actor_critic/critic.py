import torch
import torch.nn as nn

from typing import Dict
from collections import defaultdict

from utils.tools import RequiresGrad, Optimizer
from utils.networks import MLP
from utils.output_head import MLPHead
import elements

class Critic(nn.Module):
    _name = "critic"
    def __init__(self, config, device=torch.device("cpu")):
        super().__init__()
        self.config = config

        self._base = MLP(
            in_dim=(
                config.world_model.rssm.deterministic_dim
                + config.world_model.rssm.stochastic_dim * config.world_model.rssm.classes
            ),
            hidden_dim=config.critic.hidden_dim,
            hidden_layers=config.critic.hidden_layers,
            act=config.critic.act,
            use_layernorm=config.critic.use_layernorm,
            device=device,
        )

        # out layers
        self._out = MLPHead(
            in_dim=config.critic.hidden_dim,
            hidden_dim=config.critic.hidden_dim,
            hidden_layers=1,
            out_dim=1,
            output=config.critic.output,
            act=config.critic.act,
            use_layernorm=config.critic.use_layernorm,
            out_scale=config.critic.out_scale,
            device=device,
        )

        # context manager for gradient control
        self._requires_grad = RequiresGrad(self)

        # optimizer
        self._optim = Optimizer(
            name=self._name,
            parameters=self.parameters(),
            lr=config.train.optim.critic.lr,
            eps=config.train.optim.critic.eps,
            use_max_grad_norm=config.train.optim.critic.use_max_grad_norm,
            max_grad_norm=config.train.optim.critic.max_grad_norm,
        )

    def forward(self, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        deter = latent["deter"]
        stoch = latent["stoch"].flatten(-2, -1)
        x = torch.cat([deter, stoch], dim=-1)
        x = self._base(x)
        value_output = self._out(x)
        critic_outputs = {
            "value_preds": value_output.pred(),
            "value_output": value_output,
        }
        return critic_outputs

    @elements.timer.section("update_critic")
    def a2c_update(
        self,
        latent: Dict[str, torch.Tensor],
        target_returns: torch.Tensor,
        agent_mask: torch.Tensor,
    ):
        metrics = {}
        with self._requires_grad:
            value_output = self.forward(latent)["value_output"]

            loss = value_output.loss(target_returns.detach())
            loss = (loss * agent_mask).sum() / (agent_mask.sum() + 1e-5)
            optim_metric = self._optim(loss)

            # calculate mse error
            loss_mse = (value_output.pred() - target_returns) ** 2
            loss_mse = (loss_mse * agent_mask).sum() / (agent_mask.sum() + 1e-5)
            metrics.update({"critic_loss_mse": loss_mse.item()})

        metrics.update(optim_metric)
        return metrics
    
    @elements.timer.section("update_critic")
    def ppo_update(
        self,
        latent: Dict[str, torch.Tensor],
        target_returns: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ):
        if agent_mask is not None:
            target_returns = target_returns.clone()
            target_returns[agent_mask == 0] = 0
        metrics = defaultdict(float)
        with self._requires_grad:
            for _ in range(self.config.train.ppo_epochs):
                value_output = self.forward(latent)["value_output"]

                loss = value_output.loss(target_returns.detach())
                loss = loss.mean()
                optim_metric = self._optim(loss)

                # calculate mse error
                loss_mse = (value_output.pred().detach() - target_returns.detach()) ** 2
                loss_mse = loss_mse.mean()

                for key, value in optim_metric.items():
                    metrics[key] += value
                metrics["critic_loss_mse"] += loss_mse.item()

        for key, value in metrics.items():
            metrics[key] /= self.config.train.ppo_epochs
        return metrics

    def save(self):
        data = {
            "model_state_dict": self.state_dict(),
            "optim_state_dict": self._optim.state_dict(),
        }
        return data

    def load(self, data):
        self.load_state_dict(data["model_state_dict"])
        self._optim.load_state_dict(data["optim_state_dict"])
