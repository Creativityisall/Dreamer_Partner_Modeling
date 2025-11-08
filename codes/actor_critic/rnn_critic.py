import torch
import torch.nn as nn
from tensordict import TensorDict

from typing import Dict
from collections import defaultdict

from utils.tools import RequiresGrad, Optimizer
from utils.networks import MLP, CNN, GRU
from utils.output_head import MLPHead
import elements

class RNNCritic(nn.Module):
    _name = "critic"
    def __init__(self, config, obs_shape, device=torch.device("cpu")):
        super().__init__()
        self.config = config

        # enc layers
        if len(obs_shape) == 1:
            self._base = MLP(
                in_dim=obs_shape[0],
                hidden_dim=config.critic.hidden_dim,
                hidden_layers=config.critic.hidden_layers,
                act=config.critic.act,
                use_layernorm=config.critic.use_layernorm,
                use_symlog=config.critic.use_symlog,
                device=device,
            )
        elif len(obs_shape) == 3:
            self._base = CNN(
                input_shape=obs_shape,
                depth=config.critic.depth,
                mults=config.critic.mults,
                kernel=config.critic.kernel,
                act=config.critic.act,
                output_dim=config.critic.hidden_dim,
                use_layernorm=config.critic.use_layernorm,
                device=device,
            )
        else:
            raise NotImplementedError

        # rnn layers
        if config.use_rnn:
            self._rnn = GRU(
                in_dim=config.critic.hidden_dim,
                hidden_dim=config.critic.hidden_dim,
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

    def forward(
        self,
        obs: torch.Tensor,
        rnn_states_critic: torch.Tensor | None = None,
    ) -> TensorDict:
        critic_features = self._base(obs)
        if self.config.use_rnn:
            critic_features = rnn_states_critic = self._rnn(critic_features, rnn_states_critic)
        value_output = self._out(critic_features)

        critic_outputs = TensorDict(
            {
                "rnn_states_critic": rnn_states_critic,
                "value_preds": value_output.pred(),
            },
            batch_size=rnn_states_critic.shape[:-1],
        )
        return critic_outputs

    @elements.timer.section("update_critic")
    def a2c_update(self,
        obs: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        target_returns: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> Dict[str, float]:
        metrics = {}
        with self._requires_grad:
            critic_features = self._base(obs)
            if self.config.use_rnn:
                critic_features = self._rnn(critic_features, rnn_states_critic)
            value_output = self._out(critic_features)

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
    def ppo_update(self,
        obs: torch.Tensor,
        rnn_states_critic: torch.Tensor,
        target_returns: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        metrics = defaultdict(float)
        with self._requires_grad:
            for _ in range(self.config.train.ppo_epochs):
                critic_features = self._base(obs)
                if self.config.use_rnn:
                    critic_features = self._rnn(critic_features, rnn_states_critic)
                value_output = self._out(critic_features)

                loss = value_output.loss(target_returns.detach())
                loss = (loss * agent_mask).sum() / (agent_mask.sum() + 1e-5) if agent_mask is not None else loss.mean()
                optim_metric = self._optim(loss)

                # calculate mse error
                loss_mse = (value_output.pred().detach() - target_returns.detach()) ** 2
                loss_mse = (loss_mse * agent_mask).sum() / (agent_mask.sum() + 1e-5) if agent_mask is not None else loss_mse.mean()

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
