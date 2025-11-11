import torch
from tensordict.tensordict import TensorDict  # 导入 TensorDict，用于高效处理嵌套和批处理数据
import numpy as np

from typing import Dict, List
import re  # 导入正则表达式库，用于日志键匹配


class DreamerRunner:
    """
    多智能体 Dreamer/基于模型算法的执行器（Runner）。
    负责与多个并行环境进行交互、管理智能体状态、生成动作并收集数据。
    """

    def __init__(self, config, envs, actors, world_model, replay, device):
        """
        初始化执行器。

        Args:
            config (AttrDict): 训练配置。
            envs (List[MultiAgentEnv]): 并行环境实例列表。
            actors (List[Actor]): 每个智能体的演员（策略）网络列表。
            world_model (WorldModel): 世界模型实例（负责状态/信念更新）。
            replay (ReplayBuffer): 数据回放缓冲区。
            device (torch.device): 运行设备（CPU/GPU）。
        """
        self.config = config
        self.envs = envs
        self.actors = actors
        self.wm = world_model
        self.replay = replay
        self.n_agents = len(actors)  # 智能体数量 (假设 actors 列表长度等于智能体数量)
        self.device = device

    def reset(self):
        """
        重置所有环境和智能体状态，并存储第一个时间步的数据。
        """
        # 1. 初始化所有环境
        # envs.reset() 返回一个 Future，需要调用 future() 来获取结果 (兼容并行环境)
        futures = [env.reset() for env in self.envs]
        # step_dict_list: 包含每个并行环境的初始观测数据的列表 (Dict[str, np.ndarray])
        self.step_dict_list: List[Dict[str, np.ndarray]] = [future() for future in futures]

        # 2. 合并观测数据到 TensorDict
        # 将每个环境的 step_dict 转换为 TensorDict，并去除以 "log_" 开头的键
        # 然后将所有 TensorDict 沿着批次维度堆叠 (batch_size = len(self.envs))
        self.merged_step_dict: TensorDict = torch.stack(
            [
                TensorDict(
                    step_dict,
                    device=self.device,
                ).named_apply(lambda k, v: v if not k.startswith("log_") else None) for step_dict in self.step_dict_list
            ],
        )

        # 3. 初始化世界模型/智能体状态
        self.agent_state: TensorDict = self.wm.initialize_agent_state(batch_size=len(self.envs))

        # 4. 存储第一个时间步的数据到回放缓冲区
        for i, step_dict in enumerate(self.step_dict_list):
            # 添加前一时刻的动作（在 reset 之后，通常是全 0 或 No-op 动作）
            step_dict["prev_actions"] = self.agent_state["actions"][i].cpu().numpy()
            self.replay.add(step_dict, worker=i)

    @torch.no_grad()
    def step(self, agg=None, evaluation=False):
        """
        执行一个完整的环境时间步：获取动作 -> 与环境交互 -> 存储数据 -> 聚合统计。

        Args:
            agg (Aggregator, optional): 用于聚合日志统计信息的对象。
            evaluation (bool): 是否处于评估模式（会影响 Actor 的探索行为）。

        Returns:
            Tuple[int, int]: (num_steps, num_episodes) - 收集到的步骤数和完成的回合数。
        """
        # 1. 获取动作和更新智能体状态
        actor_outputs, self.agent_state = self.get_actions(
            obs=self.merged_step_dict["obs"],
            is_first=self.merged_step_dict["is_first"],
            agent_state=self.agent_state,
            # 传递可用动作信息（如果存在）
            avail_actions=(
                self.merged_step_dict["avail_actions"] if "avail_actions" in self.merged_step_dict else None),
            evaluation=evaluation,
        )

        # 2. 环境交互
        # dones 标志：当前时间步是否终止或截断 (T | D)
        dones = self.merged_step_dict["terminated"] | self.merged_step_dict["truncated"]
        actions_env = actor_outputs["actions_env"].cpu().numpy()  # 提取环境需要的动作

        # 对于未结束的环境执行 step，已结束的环境执行 reset
        futures = [
            env.step(actions_env[i]) if not dones[i]
            else env.reset()
            for i, env in enumerate(self.envs)
        ]

        # 获取环境交互结果
        self.step_dict_list: List[Dict[str, np.ndarray]] = [future() for future in futures]

        # 3. 合并新的观测数据
        self.merged_step_dict: TensorDict = torch.stack(
            [
                TensorDict(
                    step_dict,
                    device=self.device,
                ).named_apply(lambda k, v: v if not k.startswith("log_") else None) for step_dict in self.step_dict_list
            ],
        )

        # 4. 存储数据到回放缓冲区
        for i, step_dict in enumerate(self.step_dict_list):
            # 存储智能体在新时间步产生的动作，作为下一个时间步的 prev_actions
            step_dict["prev_actions"] = self.agent_state["actions"][i].cpu().numpy()
            self.replay.add(step_dict, worker=i)

        # 5. 聚合环境统计信息
        if agg is not None:
            for step_dict in self.step_dict_list:
                for key, value in step_dict.items():
                    if key.startswith("log_"):
                        # 使用正则表达式匹配，将值添加到聚合器中
                        if re.match(self.config.logging.log_keys_avg, key):
                            agg.add(key, value, agg="avg")
                        if re.match(self.config.logging.log_keys_sum, key):
                            agg.add(key, value, agg="sum")
                        if re.match(self.config.logging.log_keys_max, key):
                            agg.add(key, value, agg="max")

        # 6. 计算收集到的步骤数和完成的回合数
        num_steps = 0
        num_episodes = 0
        for step_dict in self.step_dict_list:
            done = step_dict["terminated"] or step_dict["truncated"]
            num_episodes += done  # 只要回合结束或截断，就计为一次回合完成

            # 计算步骤数时，排除某些特殊状态的步数
            if self.config.env_args.get("use_absorbing_state", False):
                # 如果使用吸收状态，则排除处于吸收状态的步数
                num_steps += not step_dict["is_trailing_absorbing_state"]
            else:
                # 否则，排除第一个时间步（reset 步）
                num_steps += not step_dict["is_first"]

        return num_steps, num_episodes

    def get_actions(
            self,
            obs: torch.Tensor,
            is_first: torch.Tensor,
            agent_state: TensorDict,
            avail_actions: torch.Tensor | None = None,
            evaluation: bool = False,
    ):
        """
        计算动作并更新世界模型中的信念状态。

        Args:
            obs (torch.Tensor): 观测值 (Batch, N_agents, Obs_dim)
            is_first (torch.Tensor): 标志，指示是否是回合的第一个时间步 (Batch, 1)
            agent_state (TensorDict): 智能体的前一时刻状态（latent, actions）
            avail_actions (torch.Tensor | None): 可用动作掩码
            evaluation (bool): 是否处于评估模式

        Returns:
            Tuple[TensorDict, TensorDict]: (actor_outputs, agent_states)
        """
        # 1. 观测编码
        embed = self.wm.encoder(obs)  # (Batch, N_agents, Embed_dim)

        # 2. 世界模型更新：根据当前观测更新信念状态
        latent: TensorDict = self.wm.observe_step(
            embed=embed,
            is_first=is_first,
            prev_actions=agent_state["actions"],
            prev_latent=agent_state["latents"],
        )  # (Batch, N_agents, Latent_dim)

        # 3. 演员（Actor）策略执行
        # 多智能体动作输出是一个列表，由各个动作组成的列表
        actor_outputs: List[TensorDict] = [
            self.actors[i](# 多智能体
                latent=latent[:, i],  # 提取并传入当前智能体的信念状态 (Batch, Latent_dim)
                # 提取并传入当前智能体的可用动作掩码
                avail_actions=avail_actions[:, i] if avail_actions is not None else None,
                evaluation=evaluation,
            )
            for i in range(len(self.actors))
        ]
        actor_outputs = torch.stack(actor_outputs, dim=1)  # 堆叠回 (Batch, N_agents, ...) 形状

        # 4. 构建新的智能体状态（用于下一个时间步）
        agent_states = TensorDict(
            {
                "latents": latent,  # 更新后的信念状态/隐状态
                "actions": actor_outputs["actions"],  # 当前时间步生成的动作
            },
            batch_size=len(self.envs),
            device=self.device,
        )

        return actor_outputs, agent_states