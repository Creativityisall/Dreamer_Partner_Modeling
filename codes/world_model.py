import torch
import torch.nn as nn
from torch.distributions import Independent, kl_divergence
from tensordict import TensorDict
import numpy as np

from typing import Dict, List, Tuple, Callable

from utils.tools import n2t, scan, RequiresGrad, Optimizer, latent_to_input, get_st_onehot
from utils.networks import MLP, GRU, CNN, Transformer
from utils.output_head import MLPHead, FigureHead
from actor_critic.actor import Actor
import elements


class MAWorldModel(nn.Module):
    _name = "world_model"

    def __init__(self, config, obs_shape, n_actions, n_agents, device):
        super().__init__()
        # 初始化基础参数
        self.config = config
        self.obs_shape = obs_shape  # 观测空间的形状
        self.n_actions = n_actions  # 动作空间维度
        self.n_agents = n_agents  # 智能体数量
        self.device = device  # 计算设备
        self.tpdv = dict(dtype=torch.float32, device=device)  # 张量参数字典

        # 世界模型组件
        # 观测编码器：将原始观测编码为特征向量
        self.encoder = ObsEncoder(config, obs_shape, device)
        # 动态模型：使用RSSM（循环状态空间模型）建模环境动态
        self.dynamics = RSSM(config, n_actions, n_agents, device)

        # 观测预测器输入维度计算
        # 确定性维度 + 随机性维度 × 类别数（如果使用分类表示）
        obs_predictor_in_dim = (
                config.world_model.rssm.deterministic_dim
                + config.world_model.rssm.stochastic_dim
                * config.world_model.rssm.classes
        )

        # 根据观测形状选择不同的观测预测器
        # 针对向量观测（1维）
        if len(obs_shape) == 1:
            self.obs_predictor = MLPHead(
                in_dim=obs_predictor_in_dim,
                hidden_dim=config.world_model.rssm.obs_predictor.hidden_dim,
                hidden_layers=config.world_model.rssm.obs_predictor.hidden_layers,
                out_dim=obs_shape[0],  # 输出维度等于观测维度
                act=config.world_model.rssm.obs_predictor.act,
                use_layernorm=config.world_model.rssm.obs_predictor.use_layernorm,
                output=config.world_model.rssm.obs_predictor.output,
                device=device,
            )
        # 针对图像观测（3维：通道×高度×宽度）
        elif len(obs_shape) == 3:
            self.obs_predictor = FigureHead(
                in_dim=obs_predictor_in_dim,
                out_shape=obs_shape,  # 输出形状与观测相同
                depth=config.world_model.rssm.obs_predictor.depth,
                mults=config.world_model.rssm.obs_predictor.mults,
                kernel=config.world_model.rssm.obs_predictor.kernel,
                act=config.world_model.rssm.obs_predictor.act,
                use_layernorm=config.world_model.rssm.obs_predictor.use_layernorm,
                device=device,
            )
        else:
            raise NotImplementedError  # 不支持其他形状的观测

        # 其他预测器组件
        # 全局智能体嵌入变换器：处理多智能体间的交互
        self.global_agent_embedding_transformer = Transformer(
            d_model=config.world_model.rssm.deterministic_dim,
            nhead=config.world_model.rssm.global_agent_embedding_transformer.nhead,
            num_layers=config.world_model.rssm.global_agent_embedding_transformer.num_layers,
            act=config.world_model.rssm.global_agent_embedding_transformer.activation,
            norm_first=config.world_model.rssm.global_agent_embedding_transformer.norm_first,
            device=device,
        )

        # 奖励预测器：预测即时奖励
        self.reward_predictor = MLPHead(
            in_dim=config.world_model.rssm.deterministic_dim,
            hidden_dim=config.world_model.rssm.reward_predictor.hidden_dim,
            hidden_layers=config.world_model.rssm.reward_predictor.hidden_layers,
            out_dim=1,  # 输出单个奖励值
            act=config.world_model.rssm.reward_predictor.act,
            use_layernorm=config.world_model.rssm.reward_predictor.use_layernorm,
            output=config.world_model.rssm.reward_predictor.output,
            out_scale=config.world_model.rssm.reward_predictor.out_scale,
            device=device,
        )

        # 终止状态预测器：预测episode是否终止（cont是continuation的缩写）
        self.cont_predictor = MLPHead(
            in_dim=config.world_model.rssm.deterministic_dim,
            hidden_dim=config.world_model.rssm.cont_predictor.hidden_dim,
            hidden_layers=config.world_model.rssm.cont_predictor.hidden_layers,
            out_dim=1,  # 输出是否继续的概率
            act=config.world_model.rssm.cont_predictor.act,
            use_layernorm=config.world_model.rssm.cont_predictor.use_layernorm,
            output=config.world_model.rssm.cont_predictor.output,
            device=device,
        ) if config.world_model.rssm.use_cont_predictor else None  # 根据配置决定是否使用

        # 动作掩码预测器：预测哪些动作是可用的
        self.act_mask_predictor = MLPHead(
            in_dim=config.world_model.rssm.deterministic_dim,
            hidden_dim=config.world_model.rssm.act_mask_predictor.hidden_dim,
            hidden_layers=config.world_model.rssm.act_mask_predictor.hidden_layers,
            out_dim=n_actions,  # 输出每个动作的可用性
            act=config.world_model.rssm.act_mask_predictor.act,
            use_layernorm=config.world_model.rssm.act_mask_predictor.use_layernorm,
            output=config.world_model.rssm.act_mask_predictor.output,
            device=device,
        ) if config.world_model.rssm.use_act_mask_predictor else None  # 根据配置决定是否使用

        # 梯度控制上下文管理器
        self._requires_grad = RequiresGrad(self)

        # 优化器
        self._optim = Optimizer(
            name=self._name,
            parameters=self.parameters(),  # 所有模型参数
            lr=config.train.optim.world_model.lr,
            eps=config.train.optim.world_model.eps,
            use_max_grad_norm=config.train.optim.world_model.use_max_grad_norm,
            max_grad_norm=config.train.optim.world_model.max_grad_norm,
        )

    def observe(
            self,
            obs: torch.Tensor | np.ndarray,
            prev_actions: torch.Tensor | np.ndarray,
            is_first: torch.Tensor | np.ndarray,
    ) -> TensorDict:
        """观测函数：处理观测序列并生成潜在状态序列"""
        # 将输入数据转换为torch张量
        obs = n2t(obs, **self.tpdv) if isinstance(obs, np.ndarray) else obs
        prev_actions = n2t(prev_actions, **self.tpdv) if isinstance(prev_actions, np.ndarray) else prev_actions
        is_first = n2t(is_first, device=self.device, dtype=torch.bool) if isinstance(is_first, np.ndarray) else is_first

        # 数据形状说明：
        # obs.shape = (时间步, 批次大小, 智能体数量, 观测形状)
        # prev_actions.shape = (时间步, 批次大小, 智能体数量, 动作维度)
        # is_first.shape = (时间步, 批次大小, 1)

        # 对观测进行编码
        embed = self.encoder(obs)

        # 使用scan函数（类似RNN的展开）逐步处理序列
        # 生成后验潜在状态序列
        post_latent: List[TensorDict] = scan(
            fn=self.observe_step,  # 每一步的处理函数
            inputs=(
                embed,
                prev_actions,
                is_first,
            ),
            init_state=None,  # 初始状态为空
        )
        return post_latent

    def prep_init_latent_for_imagination(self, samples: Dict[str, np.ndarray]) -> TensorDict:
        """为想象过程准备初始潜在状态"""
        # 将样本数据转换为torch张量
        obs = n2t(samples["obs"], **self.tpdv)
        prev_actions = n2t(samples["prev_actions"], **self.tpdv)
        rewards = n2t(samples["rewards"], **self.tpdv)
        is_first = n2t(samples["is_first"], device=self.device, dtype=torch.bool)
        agent_mask = n2t(samples["agent_mask"], **self.tpdv)
        avail_actions = n2t(samples["avail_actions"], **self.tpdv) if self.act_mask_predictor is not None else None
        terminated = n2t(samples["is_trailing_absorbing_state"], device=self.device,
                         dtype=torch.bool) if self.cont_predictor is not None else None

        # 通过观测生成后验潜在状态作为想象的初始状态
        init_latent: List[TensorDict] = self.observe(
            obs=obs,
            prev_actions=prev_actions,
            is_first=is_first,
        )
        init_latent: TensorDict = torch.stack(init_latent, dim=0)  # 将列表堆叠为张量

        # 填充初始状态的其他信息
        init_latent["agent_embeddings"] = self.global_agent_embedding_transformer(init_latent["deter"],
                                                                                  num_batch_dims=2)
        init_latent["agent_mask"] = agent_mask
        init_latent["rewards"] = rewards
        init_latent["avail_actions"] = avail_actions if avail_actions is not None else None
        init_latent["terminated"] = terminated.unsqueeze(2).repeat(1, 1, self.n_agents,
                                                                   1) if terminated is not None else None
        init_latent = init_latent.flatten(0, 1)  # 展平时间步和批次维度
        return init_latent

    @elements.timer.section("imagination")
    @torch.no_grad()  # 想象过程不需要梯度
    def imagine(
            self,
            actors: List[Actor],
            init_latent: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        生成想象轨迹：给定初始潜在状态和actor策略，生成多步想象轨迹
        """
        # 预分配张量内存
        B = init_latent["deter"].shape[0]  # 批次大小
        T = self.config.train.imagination_steps  # 想象步数
        A = self.n_agents  # 智能体数量

        # 初始化各种状态的存储张量
        agent_embeddings = torch.zeros((T + 1, B, A, self.config.world_model.rssm.deterministic_dim),
                                       device=self.device)
        deter = torch.zeros((T + 1, B, A, self.config.world_model.rssm.deterministic_dim), device=self.device)
        stoch = torch.zeros(
            (T + 1, B, A, self.config.world_model.rssm.stochastic_dim, self.config.world_model.rssm.classes),
            device=self.device)
        rewards = torch.zeros((T + 1, B, A, 1), device=self.device)
        actions_env = torch.zeros((T, B, A), device=self.device)
        terminated = torch.zeros((T + 1, B, A, 1), device=self.device)
        avail_actions = torch.zeros((T + 1, B, A, self.n_actions),
                                    device=self.device) if self.act_mask_predictor is not None else None

        # 设置初始值
        agent_embeddings[0] = init_latent["agent_embeddings"]
        deter[0] = init_latent["deter"]
        stoch[0] = init_latent["stoch"]
        rewards[0] = init_latent["rewards"]
        if self.cont_predictor is not None:
            terminated[0] = init_latent["terminated"]
        if self.act_mask_predictor is not None:
            avail_actions[0] = init_latent["avail_actions"]

        def imagine_step(step: torch.Tensor, latent: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """想象单步转换"""
            # 使用actor策略选择动作
            actor_outputs: List[TensorDict] = [
                actors[i](
                    latent=latent[:, i],  # 每个智能体的潜在状态
                    avail_actions=avail_actions[step, :, i] if self.act_mask_predictor is not None else None,
                    evaluation=False,  # 训练模式，使用探索策略
                )
                for i in range(len(actors))
            ]
            actor_outputs = torch.stack(actor_outputs, dim=1)

            # 使用动态模型想象下一步状态
            next_latent: TensorDict = self.dynamics.imagine(actor_outputs["actions"], latent)

            # 填充下一步状态信息
            actions_env[step] = actor_outputs["actions_env"]  # 环境动作
            deter[step + 1] = next_latent["deter"]  # 确定性状态
            stoch[step + 1] = next_latent["stoch"]  # 随机性状态
            agent_embeddings[step + 1] = self.global_agent_embedding_transformer(deter[step + 1])  # 全局嵌入
            rewards[step + 1] = self.reward_predictor(agent_embeddings[step + 1]).pred()  # 预测奖励
            if self.cont_predictor is not None:
                terminated[step + 1] = self.cont_predictor(agent_embeddings[step + 1]).pred()  # 预测终止状态
            if self.act_mask_predictor is not None:
                avail_actions[step + 1] = self.act_mask_predictor(agent_embeddings[step + 1]).sample()  # 预测可用动作

            return next_latent

        # 使用scan函数执行多步想象
        scan(
            fn=imagine_step,
            inputs=(torch.arange(self.config.train.imagination_steps),),  # 步数序列
            init_state=init_latent,  # 初始状态
        )

        # 整理想象轨迹数据
        imaginary_transitions: Dict[str, torch.Tensor] = {
            "agent_embeddings": agent_embeddings,
            "deter": deter,
            "stoch": stoch,
            "terminated": terminated,
            "rewards": rewards,
            "actions_env": actions_env,
        }
        if self.act_mask_predictor is not None:
            imaginary_transitions["avail_actions"] = avail_actions
        return imaginary_transitions

    def post_prior_loss(
            self,
            post: torch.Tensor,
            prior: torch.Tensor,
            loss_fn: Callable,
            transform: Callable = lambda x: x,
            free: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算后验和先验之间的损失"""
        # 计算动态损失：后验固定，先验更新
        dyn_loss = loss_fn(transform(post.detach()), transform(prior))
        dyn_loss = dyn_loss.clamp(min=free)  # 使用自由比特防止过度正则化

        # 计算表示损失：先验固定，后验更新
        rep_loss = loss_fn(transform(post), transform(prior.detach()))
        rep_loss = rep_loss.clamp(min=free)

        return dyn_loss, rep_loss

    def _latent_dist(self, logits: torch.Tensor, unimix: float = 0.0):
        """构建潜在状态的分布"""
        categorical = get_st_onehot(logits, unimix=unimix)  # 获取onehot表示的分类分布
        independent = Independent(categorical, 1)  # 创建独立分布
        return independent

    # 世界模型损失计算
    @elements.timer.section("update_world_model")
    def update(self, samples: Dict[str, np.ndarray]) -> Dict[str, float]:
        """更新世界模型参数"""
        # 转换输入数据为torch张量
        obs = n2t(samples["obs"], **self.tpdv)
        prev_actions = n2t(samples["prev_actions"], **self.tpdv)
        rewards = n2t(samples["rewards"], **self.tpdv)
        is_first = n2t(samples["is_first"], device=self.device, dtype=torch.bool)
        agent_mask = n2t(samples["agent_mask"], **self.tpdv)
        avail_actions = n2t(samples["avail_actions"], **self.tpdv) if self.act_mask_predictor is not None else None
        terminated = n2t(samples["is_trailing_absorbing_state"], device=self.device,
                         dtype=torch.bool) if self.cont_predictor is not None else None

        # 数据形状说明：
        # obs.shape = (时间步, 批次大小, 智能体数量, 观测形状)
        # prev_actions.shape = (时间步, 批次大小, 智能体数量, 动作维度)
        # rewards.shape = (时间步, 批次大小, 智能体数量, 1)
        # is_first.shape = (时间步, 批次大小, 1)
        # agent_mask.shape = (时间步, 批次大小, 智能体数量, 1)
        # avail_actions.shape = (时间步, 批次大小, 智能体数量, 动作维度)
        # terminated.shape = (时间步, 批次大小, 1)

        metrics = {}  # 存储训练指标
        loss = 0  # 总损失

        with self._requires_grad:  # 梯度控制上下文
            # 通过观测获取后验潜在状态序列
            post_latent: List[TensorDict] = self.observe(
                obs=obs,
                prev_actions=prev_actions,
                is_first=is_first,
            )

            # 移除burn-in步骤（初始的若干步用于RNN热身）
            if self.config.train.burn_in_length:
                obs = obs[self.config.train.burn_in_length:]
                rewards = rewards[self.config.train.burn_in_length:]
                is_first = is_first[self.config.train.burn_in_length:]
                agent_mask = agent_mask[self.config.train.burn_in_length:]
                post_latent = post_latent[self.config.train.burn_in_length:]
                avail_actions = avail_actions[self.config.train.burn_in_length:] if avail_actions is not None else None
                terminated = terminated[self.config.train.burn_in_length:] if terminated is not None else None

            post_latent: TensorDict = torch.stack(post_latent, dim=0)  # 堆叠为张量

            # 1. 随机潜在状态损失
            # 从确定性状态生成先验随机状态
            prior_stoch: TensorDict = self.dynamics.imagine_stoch_from_deter(post_latent["deter"], num_batch_dims=2)

            # 计算动态损失和表示损失
            dyn_loss, rep_loss = self.post_prior_loss(
                post=post_latent["logits"],
                prior=prior_stoch["logits"],
                loss_fn=kl_divergence,  # 使用KL散度
                transform=lambda x: self._latent_dist(x, unimix=self.config.world_model.rssm.unimix),
                free=self.config.train.free_bits,  # 自由比特参数
            )

            # 形状检查
            assert dyn_loss.unsqueeze(-1).shape == agent_mask.shape, (dyn_loss.shape, agent_mask.shape)
            assert rep_loss.unsqueeze(-1).shape == agent_mask.shape, (rep_loss.shape, agent_mask.shape)

            # 计算均值并加权加到总损失中
            dyn_loss = dyn_loss.mean()
            rep_loss = rep_loss.mean()
            loss += self.config.train.stoch_dyn_scale * dyn_loss
            loss += self.config.train.stoch_rep_scale * rep_loss
            metrics.update({"dyn_loss": dyn_loss.item(), "rep_loss": rep_loss.item()})

            # 记录后验和先验的熵
            prior_dist = self._latent_dist(prior_stoch["logits"], unimix=self.config.world_model.rssm.unimix)
            post_dist = self._latent_dist(post_latent["logits"], unimix=self.config.world_model.rssm.unimix)
            prior_entropy = prior_dist.entropy().mean()
            post_entropy = post_dist.entropy().mean()

            metrics.update(
                {
                    "post_stoch_entropy": post_entropy.item(),
                    "prior_stoch_entropy": prior_entropy.item(),
                }
            )

            # 准备预测器输入
            predictor_inputs = latent_to_input(post_latent)

            # 2. 观测预测损失
            obs_output = self.obs_predictor(predictor_inputs)
            obs_loss = obs_output.loss(obs)
            # 对多余维度求和并保持形状一致
            obs_loss = obs_loss.sum(dim=list(range(3, obs.ndim))).unsqueeze(-1)
            assert obs_loss.shape == agent_mask.shape, (obs_loss.shape, agent_mask.shape)
            obs_loss = obs_loss.mean()
            loss += self.config.train.obs_scale * obs_loss
            metrics.update({"obs_loss": obs_loss.item()})

            # 生成智能体的全局嵌入
            agent_embeddings = self.global_agent_embedding_transformer(post_latent["deter"], num_batch_dims=2)

            # 创建非首步掩码（首步通常用于初始化，不用于训练）
            not_first = ~is_first.unsqueeze(-2).repeat(1, 1, self.n_agents, 1)

            # 3. 连续状态预测损失
            if self.config.world_model.rssm.use_cont_predictor:
                # 根据配置决定是否使用特征梯度
                if self.config.world_model.rssm.cont_predictor.enable_feat_grad:
                    cont_output = self.cont_predictor(agent_embeddings)
                else:
                    cont_output = self.cont_predictor(agent_embeddings.detach())

                # 调整终止状态形状以匹配掩码
                terminated = terminated.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
                assert terminated.shape == agent_mask.shape, (terminated.shape, agent_mask.shape)

                # 计算损失（只对非首步计算）
                cont_loss = cont_output.loss(terminated.float())
                cont_loss = (cont_loss * not_first).sum() / (not_first.sum() + 1e-5)
                loss += self.config.train.cont_scale * cont_loss
                metrics.update({"cont_loss": cont_loss.item()})

                # 计算准确率
                cont_acc = (cont_output.mode == terminated).float()
                cont_acc = (cont_acc * not_first).sum() / (not_first.sum() + 1e-5)
                metrics.update({"cont_acc": cont_acc.item()})

            # 4. 奖励预测损失
            if self.config.world_model.rssm.reward_predictor.enable_feat_grad:
                reward_output = self.reward_predictor(agent_embeddings)
            else:
                reward_output = self.reward_predictor(agent_embeddings.detach())

            reward_loss = reward_output.loss(rewards)
            reward_loss = (reward_loss * not_first).sum() / (not_first.sum() + 1e-5)
            loss += self.config.train.reward_scale * reward_loss
            metrics.update({"reward_loss": reward_loss.item()})

            # 记录均方奖励误差
            reward_preds = reward_output.pred()
            assert reward_preds.shape == rewards.shape, (reward_preds.shape, rewards.shape)
            reward_loss_mse = (reward_preds - rewards) ** 2
            reward_loss_mse = (reward_loss_mse * not_first).sum() / (not_first.sum() + 1e-5)
            metrics.update({"reward_loss_mse": reward_loss_mse.item()})

            # 5. 动作掩码预测损失
            if self.config.world_model.rssm.use_act_mask_predictor:
                if self.config.world_model.rssm.act_mask_predictor.enable_feat_grad:
                    act_mask_output = self.act_mask_predictor(agent_embeddings)
                else:
                    act_mask_output = self.act_mask_predictor(agent_embeddings.detach())

                act_mask_loss = act_mask_output.loss(avail_actions).sum(dim=-1, keepdim=True)
                act_mask_loss = (act_mask_loss * not_first).sum() / (not_first.sum() + 1e-5)
                loss += self.config.train.act_mask_scale * act_mask_loss
                metrics.update({"act_mask_loss": act_mask_loss.item()})

                # 计算动作掩码准确率
                act_mask_acc = (act_mask_output.mode == avail_actions).float().all(dim=-1, keepdim=True)
                act_mask_acc = (act_mask_acc * not_first).sum() / (not_first.sum() + 1e-5)
                metrics.update({"act_mask_acc": act_mask_acc.item()})

            # 执行优化步骤
            optim_metric = self._optim(loss)

        # 更新指标字典
        metrics.update(optim_metric)
        return metrics

    def initialize_agent_state(self, batch_size: int) -> TensorDict:
        """初始化智能体状态"""
        agent_states = TensorDict(
            {
                "latents": {
                    "deter": torch.zeros(  # 确定性状态
                        batch_size,
                        self.n_agents,
                        self.config.world_model.rssm.deterministic_dim,
                    ),
                    "stoch": torch.zeros(  # 随机性状态
                        batch_size,
                        self.n_agents,
                        self.config.world_model.rssm.stochastic_dim,
                        self.config.world_model.rssm.classes,
                    ),
                    "logits": torch.zeros(  # 分类logits
                        batch_size,
                        self.n_agents,
                        self.config.world_model.rssm.stochastic_dim,
                        self.config.world_model.rssm.classes,
                    ),
                },
                "actions": torch.zeros(  # 动作
                    batch_size,
                    self.n_agents,
                    self.n_actions,
                ),
            },
            batch_size=(batch_size, self.n_agents),
            device=self.device,
        )
        return agent_states

    def observe_step(
            self,
            embed: torch.Tensor,
            prev_actions: torch.Tensor,
            is_first: torch.Tensor,
            prev_latent: TensorDict | None,
    ) -> TensorDict:
        """单步观测处理：处理单个时间步的观测更新潜在状态"""
        # 初始化或重置状态
        if prev_latent is None:
            # 首次调用，初始化所有状态
            agent_states = self.initialize_agent_state(batch_size=embed.size(0))
            prev_latent = {
                "deter": agent_states["latents"]["deter"],
                "stoch": agent_states["latents"]["stoch"],
            }
            prev_actions = agent_states["actions"]
        elif is_first.any():
            # 有episode开始，重置对应批次的状态
            agent_states = self.initialize_agent_state(batch_size=is_first.sum().item())
            prev_latent["deter"][is_first.squeeze(-1)] = agent_states["latents"]["deter"]
            prev_latent["stoch"][is_first.squeeze(-1)] = agent_states["latents"]["stoch"]
            prev_actions = prev_actions.clone()
            prev_actions[is_first.squeeze(-1)] = agent_states["actions"]

        # 使用动态模型更新潜在状态
        latent: TensorDict = self.dynamics.observe(
            embed=embed,
            prev_actions=prev_actions,
            prev_latent=prev_latent,
        )
        return latent

    def save(self):
        """保存模型状态"""
        data = {
            "model_state_dict": self.state_dict(),  # 模型参数
            "optim_state_dict": self._optim.state_dict(),  # 优化器状态
        }
        return data

    def load(self, data):
        """加载模型状态"""
        self.load_state_dict(data["model_state_dict"])  # 加载模型参数
        self._optim.load_state_dict(data["optim_state_dict"])  # 加载优化器状态

class RSSM(nn.Module):
    def __init__(self, config, n_actions, n_agents, device):
        super().__init__()
        self.config = config
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = device

        self.tot_stoch_dim = config.world_model.rssm.stochastic_dim * config.world_model.rssm.classes
        self._mlp_stoch = MLP(
            in_dim=self.tot_stoch_dim,
            hidden_dim=config.world_model.rssm.mlp.hidden_dim,
            hidden_layers=config.world_model.rssm.mlp.hidden_layers,
            act=config.world_model.rssm.mlp.act,
            use_layernorm=config.world_model.rssm.mlp.use_layernorm,
            device=device,
        )
        self._mlp_action = MLP(
            in_dim=n_actions,
            hidden_dim=config.world_model.rssm.mlp.hidden_dim,
            hidden_layers=config.world_model.rssm.mlp.hidden_layers,
            act=config.world_model.rssm.mlp.act,
            use_layernorm=config.world_model.rssm.mlp.use_layernorm,
            device=device,
        )

        self._rnn = GRU(
            in_dim=2 * config.world_model.rssm.mlp.hidden_dim,
            hidden_dim=config.world_model.rssm.deterministic_dim,
            use_layernorm=config.world_model.rssm.rnn.use_layernorm,
            device=device,
        )

        self._obs_stoch_logits = MLP(
            in_dim=config.world_model.rssm.deterministic_dim+config.world_model.encoder.hidden_dim,
            hidden_dim=config.world_model.rssm.hidden_dim,
            hidden_layers=config.world_model.rssm.obs_layers,
            out_dim=config.world_model.rssm.stochastic_dim * config.world_model.rssm.classes,
            act=config.world_model.rssm.act,
            use_layernorm=config.world_model.rssm.use_layernorm,
            device=device,
        )
        # 功能是从确定性状态预测得到随机状态，随机状态的对数向量分两种情况预测，一种是Transformer，一种是使用MLP
        # 其随机状态的随机性就体现在这个使用对数概率的形式上
        if config.world_model.rssm.use_img_stoch_transformer:
            self._img_stoch_logits = Transformer(
                d_model=config.world_model.rssm.deterministic_dim,
                out_dim=config.world_model.rssm.stochastic_dim * config.world_model.rssm.classes,
                nhead=config.world_model.rssm.img_stoch_transformer.nhead,
                num_layers=config.world_model.rssm.img_stoch_transformer.num_layers,
                act=config.world_model.rssm.img_stoch_transformer.activation,
                norm_first=config.world_model.rssm.img_stoch_transformer.norm_first,
                device=device,
            )
        else:
            self._img_stoch_logits = MLP(
                in_dim=config.world_model.rssm.deterministic_dim,
                hidden_dim=config.world_model.rssm.hidden_dim,
                hidden_layers=config.world_model.rssm.obs_layers,
                out_dim=config.world_model.rssm.stochastic_dim * config.world_model.rssm.classes,
                act=config.world_model.rssm.act,
                use_layernorm=config.world_model.rssm.use_layernorm,
                device=device,
            )

    def observe(
            self,
            embed: torch.Tensor,
            prev_actions: torch.Tensor,
            prev_latent: TensorDict | Dict[str, torch.Tensor],
        ) -> TensorDict:
        # 1. generate determinisitic embedding
        prev_deter = prev_latent["deter"]
        prev_stoch = prev_latent["stoch"]

        x1 = self._mlp_stoch(
            prev_stoch.reshape(-1, self.tot_stoch_dim)
        ).reshape(*prev_stoch.shape[:-2], -1)
        x2 = self._mlp_action(prev_actions)
        x = torch.cat([x1, x2], dim=-1)
        deter = self._rnn(x, prev_deter)

        # 2. generate stochastic embedding
        input = torch.cat([embed, deter], dim=-1)
        stoch_logits = self._obs_stoch_logits(input)
        stoch_logits = stoch_logits.reshape(
            *input.shape[:-1],
            self.config.world_model.rssm.stochastic_dim,
            self.config.world_model.rssm.classes,
        )
        # shoch_logits.shape = (batch_size, n_agents, n_stochastic_dim, n_classes)
        stoch = get_st_onehot(stoch_logits, unimix=self.config.world_model.rssm.unimix).rsample()

        latent = TensorDict(
            {
                "deter": deter,
                "logits": stoch_logits,
                "stoch": stoch,
            },
            batch_size=deter.shape[:-1],
            device=self.device,
        )
        return latent

    def imagine(
            self,
            prev_actions: torch.Tensor,
            prev_latent: Dict[str, torch.Tensor],
        ) -> TensorDict:
        # 1. generate deterministic embedding at next timestep
        prev_deter = prev_latent["deter"]
        prev_stoch = prev_latent["stoch"]

        x1 = self._mlp_stoch(
            prev_stoch.reshape(-1, self.tot_stoch_dim)
        ).reshape(*prev_stoch.shape[:-2], -1)
        x2 = self._mlp_action(prev_actions)
        x = torch.cat([x1, x2], dim=-1)
        deter = self._rnn(x, prev_deter)

        # 2. generate stochastic embedding at next timestep
        prior_stoch = self.imagine_stoch_from_deter(deter, num_batch_dims=1)

        latent = TensorDict(
            {
                "deter": deter,
                "logits": prior_stoch["logits"],
                "stoch": prior_stoch["stoch"],
            },
            batch_size=deter.shape[:-1],
            device=self.device,
        )
        return latent

    def imagine_stoch_from_deter(
        self,
        deter: torch.Tensor,
        num_batch_dims: int = 1,
    ) -> TensorDict:
        """
        Args:
            deter: shape = (ts, bs, n_agents, n_deterministic_dim)
        """
        if self.config.world_model.rssm.use_img_stoch_transformer:
            stoch_logits = self._img_stoch_logits(deter, num_batch_dims=num_batch_dims)
        else:
            stoch_logits = self._img_stoch_logits(deter)
        stoch_logits = stoch_logits.reshape(
            *deter.shape[:-1],
            self.config.world_model.rssm.stochastic_dim,
            self.config.world_model.rssm.classes,
        )
        # stoch_logits.shape = (ts, bs, n_agents, n_stochastic_dim, n_classes)

        prior = TensorDict(
            {
                "logits": stoch_logits,
                "stoch": get_st_onehot(stoch_logits, unimix=self.config.world_model.rssm.unimix).rsample(), # re-parametrization trick
            },
            batch_size=deter.shape[:-1],
            device=self.device,
        )
        return prior

# 注意所有的obs都需要经过这个encoder的处理得到特征向量
class ObsEncoder(nn.Module):
    def __init__(self, config, obs_shape, device):
        super().__init__()
        self.config = config
        self.obs_shape = obs_shape
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)

        # 针对一般的obs编码
        if len(obs_shape) == 1:
            obs_size = obs_shape[0]
            self._base = MLP(
                in_dim=obs_size,
                hidden_dim=config.world_model.encoder.hidden_dim,
                hidden_layers=config.world_model.encoder.hidden_layers,
                act=config.world_model.encoder.act,
                use_layernorm=config.world_model.encoder.use_layernorm,
                use_symlog=config.world_model.encoder.use_symlog,
                device=device,
            )
        # 针对图像的obs编码
        elif len(obs_shape) == 3:
            self._base = CNN(
                input_shape=obs_shape,
                depth=config.world_model.encoder.depth,
                mults=config.world_model.encoder.mults,
                kernel=config.world_model.encoder.kernel,
                act=config.world_model.encoder.act,
                output_dim=config.world_model.encoder.hidden_dim,
                use_layernorm=config.world_model.encoder.use_layernorm,
                device=device,
            )
        else:
            raise ValueError(f"Observation shape {obs_shape} not supported")

    def forward(self, obs: torch.Tensor):
        return self._base(obs)
