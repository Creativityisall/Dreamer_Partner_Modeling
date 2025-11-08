import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensordict import TensorDict

from typing import Dict
from collections import defaultdict

from utils.tools import Optimizer, RequiresGrad
from utils.networks import MLP, CNN, GRU, weight_init
import elements

class RNNActor(nn.Module):
    _name = "actor"
    def __init__(self, config, obs_shape, n_agents, n_actions, device=torch.device("cpu")):
        super().__init__()
        self.config = config
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.device = device

        # encoder layers
        if len(obs_shape) == 1:
            self._base = MLP(
                in_dim=obs_shape[0],
                hidden_dim=config.actor.hidden_dim,
                hidden_layers=config.actor.hidden_layers,
                act=config.actor.act,
                use_layernorm=config.actor.use_layernorm,
                use_symlog=config.actor.use_symlog,
                device=device,
            )
        elif len(obs_shape) == 3:
            self._base = CNN(
                input_shape=obs_shape,
                depth=config.actor.depth,
                mults=config.actor.mults,
                kernel=config.actor.kernel,
                act=config.actor.act,
                output_dim=config.actor.hidden_dim,
                use_layernorm=config.actor.use_layernorm,
                device=device,
            )
        else:
            raise NotImplementedError

        # rnn layers
        if config.use_rnn:
            self._rnn = GRU(
                in_dim=config.actor.hidden_dim,
                hidden_dim=config.actor.hidden_dim,
                use_layernorm=config.actor.use_layernorm,
                device=device,
            )

        # act layers
        self._out = nn.Linear(config.actor.hidden_dim, n_actions, device=device)
        weight_init(self._out, scale=config.actor.out_scale)

        # context manager for gradient control
        self._requires_grad = RequiresGrad(self)

        # optimizer
        self._optim = Optimizer(
            name=self._name,
            parameters=self.parameters(),
            lr=config.train.optim.actor.lr,
            eps=config.train.optim.actor.eps,
            use_max_grad_norm=config.train.optim.actor.use_max_grad_norm,
            max_grad_norm=config.train.optim.actor.max_grad_norm,
        )

    def forward(
            self,
            obs: torch.Tensor,
            rnn_states: torch.Tensor | None = None,
            avail_actions: torch.Tensor | None = None,
            evaluation: bool = False,
        ) -> Dict[str, torch.Tensor]:
        actor_outputs = {}

        # 先过编码层再过RNN。
        actor_features = self._base(obs)
        if self.config.use_rnn:
            assert rnn_states is not None
            actor_features = rnn_states = self._rnn(actor_features, rnn_states)
        logits = self._out(actor_features)
        if avail_actions is not None:
            logits[avail_actions == 0] = -1e8
        actions_dist = Categorical(logits=logits)
        if evaluation:
            actions = actions_dist.mode
        else:
            actions = actions_dist.sample()

        actor_outputs = TensorDict(
            {
                "rnn_states": rnn_states,
                "actions_env": actions,
                "actions": F.one_hot(actions, num_classes=self.n_actions).float(),
                "logits": logits,
                "log_probs": F.log_softmax(logits, dim=-1),
                "entropy": actions_dist.entropy(),
            },
            batch_size=rnn_states.shape[:-1],
            device=self.device,
        )
        return actor_outputs

    @elements.timer.section("update_actor")
    def a2c_update(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        actions_env: torch.Tensor,
        agent_mask: torch.Tensor,
        advantages: torch.Tensor,
        avail_actions: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        metrics = {}
        with self._requires_grad:
            actor_outputs = self.forward(obs, rnn_states, avail_actions)

            # policy gradient loss
            action_log_probs = actor_outputs["log_probs"]
            action_log_probs = action_log_probs.gather(dim=-1, index=actions_env.long().unsqueeze(-1))
            policy_loss = - action_log_probs * advantages
            policy_loss = (policy_loss * agent_mask).sum() / (agent_mask.sum() + 1e-5)

            # entropy loss
            entropy = actor_outputs["entropy"].unsqueeze(-1)
            entropy = (entropy * agent_mask).sum() / (agent_mask.sum() + 1e-5)
            entropy_loss = - entropy

            loss = policy_loss + self.config.train.entropy_coef * entropy_loss
            optim_metric = self._optim(loss)

        for key, value in optim_metric.items():
            metrics[key] = value
        metrics["policy_loss"] = policy_loss.item()
        metrics["entropy"] = entropy.item()
        return metrics

    @elements.timer.section("update_actor")
    def ppo_update(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        advantages: torch.Tensor,
        actions_env: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        avail_actions: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        actor_outputs = self.forward(obs, rnn_states, avail_actions)
        old_action_log_probs = actor_outputs["log_probs"].detach()
        old_action_log_probs = old_action_log_probs.gather(dim=-1, index=actions_env.long().unsqueeze(-1))

        metrics = defaultdict(float)
        with self._requires_grad:
            for _ in range(self.config.train.ppo_epochs):
                actor_outputs = self.forward(obs, rnn_states, avail_actions)

                # importance weight
                action_log_probs = actor_outputs["log_probs"]
                action_log_probs = action_log_probs.gather(dim=-1, index=actions_env.long().unsqueeze(-1))
                imp_weights = torch.exp(action_log_probs - old_action_log_probs)
                assert imp_weights.shape == advantages.shape, (imp_weights.shape, advantages.shape)
                surr1 = imp_weights * advantages
                surr2 = torch.clamp(imp_weights, 1 - self.config.train.clip_param, 1 + self.config.train.clip_param) * advantages
                surr = torch.min(surr1, surr2)
                policy_loss = - (surr * agent_mask).sum() / (agent_mask.sum() + 1e-5) if agent_mask is not None else - surr.mean()

                # entropy loss
                entropy = actor_outputs["entropy"].unsqueeze(-1)
                entropy = (entropy * agent_mask).sum() / (agent_mask.sum() + 1e-5) if agent_mask is not None else entropy.mean()
                entropy_loss = - entropy

                loss = policy_loss + self.config.train.entropy_coef * entropy_loss
                optim_metric = self._optim(loss)

                for key, value in optim_metric.items():
                    metrics[key] += value
                metrics["policy_loss"] += policy_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["ratio"] += imp_weights.mean().item()

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
