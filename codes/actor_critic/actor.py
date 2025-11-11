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

class Actor(nn.Module):
    _name = "actor"
    def __init__(self, config, n_agents, n_actions, agent_index=0, device=torch.device("cpu")):# TODO:增加agent的序号，后面编码的时候会按照这个顺序进行。
        super().__init__()
        self.config = config
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_index = agent_index
        self.partner_hidden_dim = config.partner_model.hidden_dim
        self.device = device

        self._base = MLP(
            in_dim=(
                config.world_model.rssm.deterministic_dim
                + config.world_model.rssm.stochastic_dim * config.world_model.rssm.classes # 这里的参数表达的是这是输出的离散类型随机变量经过one-hot后的结果。
            ),                                                                             # 每个随机变量有stochastic_dim，每个维度可能是classes个整数中的一个
            hidden_dim=config.actor.hidden_dim,
            hidden_layers=config.actor.hidden_layers,
            act=config.actor.act,
            use_layernorm=config.actor.use_layernorm,
            device=device,
        )

        # partner modeling
        self.partner_model = Partner_Model(# TODO: to confirm the dimensions
            in_dim=(
                config.world_model.rssm.deterministic_dim
                + config.world_model.rssm.stochastic_dim * config.world_model.rssm.classes
            ),
            hidden_dim=config.partner_model.hidden_dim,
            hidden_layers=config.partner_model.hidden_layers,
            out_dim = config.partner_model.hidden_dim * (n_agents - 1), #TODO:确认
            act=config.partner_model.act,
            use_layernorm=config.partner_model.use_layernorm,
            device=device
        )
        self.linear = nn.Linear(config.partner_model.hidden_dim * n_agents, config.partner_model.hidden_dim * n_agents) #TODO:这里输出采用什么维度?
        weight_init(self.linear)

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
            latent: Dict[str, torch.Tensor],
            avail_actions: torch.Tensor | None = None,
            evaluation: bool = False,
        ) -> TensorDict:
        deter = latent["deter"]
        stoch = latent["stoch"].flatten(-2, -1)
        x = torch.cat([deter, stoch], dim=-1)

        # 这里先不要使用分布，输出logits，在更新的时候计算分布
        partner_vector = self.partner_model(x)

        x = self._base(x)

        # 这里需要总共的agent数量以及agent序号。
        x = torch.cat([partner_vector[:self.partner_hidden_dim * self.agent_index], x, partner_vector[-self.partner_hidden_dim * (self.n_agents - 1 - self.agent_index):]], dim=-1)
        local_modeling_vector = self.linear(x)

        logits = self._out(local_modeling_vector)
        if avail_actions is not None:
            logits[avail_actions == 0] = -1e8
        actions_dist = Categorical(logits=logits)
        if evaluation:
            actions = actions_dist.mode # 返回这个分布中的众数，即其中概率最大的动作
        else:
            actions = actions_dist.sample()

        actor_outputs = TensorDict(
            {
                "actions_env": actions,
                "actions": F.one_hot(actions, num_classes=self.n_actions).float(),
                "logits": logits,
                "log_probs": F.log_softmax(logits, dim=-1),
                "entropy": actions_dist.entropy(),
                "partner_vector": local_modeling_vector,
            },
            batch_size=deter.shape[:-1],
            device=self.device,
        )
        return actor_outputs

    # TODO:需要传入队友建模的向量以及全局输出进行对齐。
    @elements.timer.section("update_actor")
    def a2c_update(
        self,
        latent: Dict[str, torch.Tensor],
        advantages: torch.Tensor,
        actions_env: torch.Tensor,
        agent_mask: torch.Tensor,
        avail_actions: torch.Tensor | None = None,
        global_vectors: torch.Tensor | None = None,
        local_vectors: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        metrics = {}
        with self._requires_grad:
            actor_outputs = self.forward(latent, avail_actions)

            # policy gradient loss
            action_log_probs = actor_outputs["log_probs"]
            action_log_probs = action_log_probs.gather(dim=-1, index=actions_env.long().unsqueeze(-1))
            policy_loss = - action_log_probs * advantages
            policy_loss = (policy_loss * agent_mask).sum() / (agent_mask.sum() + 1e-5)

            # entropy loss
            entropy = actor_outputs["entropy"].unsqueeze(-1)
            entropy = (entropy * agent_mask).sum() / (agent_mask.sum() + 1e-5)
            entropy_loss = - entropy

            # TODO:partner_model loss,这里只更新局部的partner_modeling
            # 目前的实现是特征向量，暂时使用MSE，后续可以尝试KL散度
            A = local_vectors.shape[1]
            expended_global_vectors = global_vectors.unsqueeze(1).expand(-1, A, -1).detach()
            mse_loss_fn = nn.MSELoss(reduction='mean')
            local_partner_loss = mse_loss_fn(local_vectors * agent_mask, expended_global_vectors * agent_mask)

            loss = policy_loss + self.config.train.entropy_coef * entropy_loss + self.config.train.local_partner_loss_coef * local_partner_loss
            optim_metric = self._optim(loss)

        for key, value in optim_metric.items():
            metrics[key] = value
        metrics["policy_loss"] = policy_loss.item()
        metrics["entropy"] = entropy.item()
        return metrics

    @elements.timer.section("update_actor")
    def ppo_update(
        self,
        latent: Dict[str, torch.Tensor],
        advantages: torch.Tensor,
        actions_env: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        avail_actions: torch.Tensor | None = None,
        global_vectors: torch.Tensor | None = None,
        local_vectors: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        actor_outputs = self.forward(latent, avail_actions)
        old_action_log_probs = actor_outputs["log_probs"].detach()
        old_action_log_probs = old_action_log_probs.gather(dim=-1, index=actions_env.long().unsqueeze(-1))

        metrics = defaultdict(float)
        with self._requires_grad:
            for _ in range(self.config.train.ppo_epochs):
                actor_outputs = self.forward(latent, avail_actions)

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

                # TODO:partner_model loss,这里只更新局部的partner_modeling
                A = local_vectors.shape[1]
                expended_global_vectors = global_vectors.unsqueeze(1).expand(-1, A, -1).detach()
                mse_loss_fn = nn.MSELoss(reduction='mean')
                local_partner_loss = mse_loss_fn(local_vectors * agent_mask, expended_global_vectors * agent_mask)

                loss = policy_loss + self.config.train.entropy_coef * entropy_loss + self.config.train.local_partner_loss_coef * local_partner_loss

                optim_metric = self._optim(loss)

                for key, value in optim_metric.items():
                    metrics[key] += value
                metrics["policy_loss"] += policy_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["ratio"] += imp_weights.mean().item()
                metrics["local_partner_loss"] += local_partner_loss.item()# TODO

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
