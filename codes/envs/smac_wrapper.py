from smac.env import StarCraft2Env  # 导入 SMAC 的 StarCraft II 环境
import numpy as np

from typing import List

from .multiagentenv import MultiAgentEnv  # 导入抽象基类 MultiAgentEnv


class SMACWrapper(MultiAgentEnv):
    """
    StarCraft II Multi-Agent Challenge (SMAC) 环境的封装器。

    主要功能是将 SMAC 的接口适配到 MultiAgentEnv，并实现吸收状态逻辑。
    """

    def __init__(
            self,
            map_name,
            use_absorbing_state,
            trailing_absorbing_state_length,
            seed,
            **kwargs
    ):
        # 初始化 StarCraft2Env 实例
        self.env = StarCraft2Env(map_name=map_name, seed=seed, **kwargs)
        env_info = self.env.get_env_info()  # 获取环境信息

        # 设置环境参数
        self.n_actions = env_info["n_actions"]  # 单个智能体的总动作数
        self.n_agents = env_info["n_agents"]  # 智能体数量
        self.obs_shape = (env_info["obs_shape"],)  # 观测维度
        self.episode_limit = self.env.episode_limit  # 回合最大步数

        # --- 吸收状态 (Absorbing State) 相关参数初始化 ---
        self.use_absorbing_state = use_absorbing_state
        self.trailing_absorbing_state_length = trailing_absorbing_state_length  # 吸收状态持续步数

        # 如果使用吸收状态，则持续步数必须大于 0
        assert not self.use_absorbing_state or self.trailing_absorbing_state_length > 0

        self.is_trailing_absorbing_state = False  # 标志：当前是否处于吸收状态
        self.cur_absorbing_state_length = -1  # 当前已持续的吸收状态步数

        # 定义吸收状态下的观测、动作和奖励等
        self.absorbing_obs = np.zeros(self.obs_shape, dtype=np.float32)
        self.absorbing_avail_actions = [0] * self.n_actions
        self.absorbing_avail_actions[0] = 1  # 吸收状态下只允许 NO-OP (动作 0)
        self.absorbing_agent_mask = np.zeros((self.n_agents, 1))  # 吸收状态下智能体被视为“死亡”或不活跃
        self.absorbing_rewards = np.zeros((self.n_agents, 1))  # 吸收状态下奖励为 0

    def step(self, actions: List[int]):
        """
        执行环境的一个时间步，包含吸收状态逻辑。

        Args:
            actions: 所有智能体的动作列表。

        Returns:
            dict: 包含下一步观测、奖励、终止等信息的字典。
        """
        result = {}

        # --- 1. 正常环境执行阶段 ---
        if not self.is_trailing_absorbing_state:
            # 执行 SMAC 步骤
            rews, terminated, self.info = self.env.step(actions)

            # 检查是否进入吸收状态：如果环境终止且开启了吸收状态
            if self.use_absorbing_state and terminated:
                self.is_trailing_absorbing_state = True
                self.cur_absorbing_state_length = 0
                terminated = False  # 在吸收状态持续期间，环境逻辑上未“终止”

            # 获取观测、可用动作和智能体掩码
            obss = self.get_obs()
            avail_actions = self.get_avail_actions()
            agent_mask = self.get_agent_mask()
            # 将奖励形状调整为 (n_agents, 1)，并复制（SMAC通常返回总奖励，这里假设它被应用于所有智能体）
            rewards = np.ones((self.n_agents, 1), dtype=np.float32) * rews

        # --- 2. 吸收状态持续阶段 ---
        else:
            self.cur_absorbing_state_length += 1
            # 检查是否达到吸收状态的持续长度限制
            terminated = (self.cur_absorbing_state_length >= self.trailing_absorbing_state_length)

            # 使用预定义的吸收状态数据
            obss = [self.absorbing_obs] * self.n_agents
            avail_actions = [self.absorbing_avail_actions] * self.n_agents
            rewards = self.absorbing_rewards
            agent_mask = self.absorbing_agent_mask

        # --- 3. 构建结果字典 ---
        result = {
            "obs": np.stack(obss, axis=0),
            "avail_actions": np.stack(avail_actions, axis=0),
            # 注意：这里 agent_mask 采用 np.stack 是为了匹配 SMACWrapper 的 get_agent_mask() 输出形状
            "agent_mask": agent_mask,
            "rewards": rewards,
            "terminated": np.array([terminated], dtype=np.bool),
            "truncated": np.array([False], dtype=np.bool),  # SMAC 通常由 terminated 标志处理，无需单独的 truncated
            "is_first": np.array([False], dtype=np.bool),
            "is_trailing_absorbing_state": np.array([self.is_trailing_absorbing_state], dtype=np.bool),
        }

        # 如果最终回合结束（terminated），记录是否获胜
        if terminated:
            result["log_battle_won"] = self.info.get("battle_won", False)

        return result

    def get_obs(self) -> List[np.ndarray]:
        """
        返回所有智能体的当前观测值列表。

        如果智能体已死亡，则返回吸收状态观测值 (zero-padding)。
        """
        agents_obs = []
        for i in range(self.n_agents):
            # SMAC 的 death_tracker_ally[i] == 1 表示智能体 i 已死亡
            if self.env.death_tracker_ally[i] == 1:
                agents_obs.append(self.absorbing_obs)
            else:
                agents_obs.append(self.env.get_obs_agent(i))
        return agents_obs

    def get_obs_size(self):
        """返回单个智能体观测值的维度大小。"""
        return self.env.get_obs_size()

    def get_state(self):
        """返回环境的全局状态。"""
        return self.env.get_state()

    def get_state_size(self):
        """返回全局状态的维度大小。"""
        return self.env.get_state_size()

    def get_avail_actions(self) -> List[np.ndarray]:
        """
        返回所有智能体当前可用的动作掩码。

        如果智能体已死亡，则只允许执行 NO-OP 动作。
        """
        avail_actions = []
        for agent_id in range(self.env.n_agents):
            if self.env.death_tracker_ally[agent_id] == 1:
                avail_agent = self.absorbing_avail_actions
            else:
                avail_agent = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_total_actions(self):
        """返回单个智能体的总动作数。"""
        return self.env.get_total_actions()

    def get_agent_mask(self):
        """
        返回智能体掩码：1 表示活跃，0 表示已死亡。

        Returns:
            np.ndarray: 形状为 (n_agents, 1) 的掩码。
        """
        # death_tracker_ally 中 1 表示死亡，0 表示存活
        # 1 - death_tracker_ally 得到：0 表示死亡，1 表示存活 (即掩码)
        agent_mask = 1 - self.env.death_tracker_ally[..., None]
        return agent_mask

    def reset(self, seed=None, options=None):
        """
        重置环境，并返回初始观测和信息。
        """
        if seed is not None:
            self.env.seed(seed)
        self.env.reset()

        # 获取初始观测和可用动作
        obss = self.get_obs()
        avail_actions = self.get_avail_actions()

        # 重置吸收状态标志
        if self.use_absorbing_state:
            self.is_trailing_absorbing_state = False
            self.cur_absorbing_state_length = -1

        # 构建初始结果字典
        result = {
            "obs": np.stack(obss, axis=0),
            "avail_actions": np.stack(avail_actions, axis=0),
            "agent_mask": self.get_agent_mask(),
            "rewards": np.zeros((self.n_agents, 1), dtype=np.float32),
            "terminated": np.array([False], dtype=np.bool),
            "truncated": np.array([False], dtype=np.bool),
            "is_first": np.array([True], dtype=np.bool),
            "is_trailing_absorbing_state": np.array([False], dtype=np.bool),
        }
        return result

    def render(self):
        """渲染环境。"""
        self.env.render()

    def close(self):
        """关闭环境。"""
        self.env.close()

    def seed(self, seed=None):
        """设置环境种子。"""
        self.env._seed = seed

    def save_replay(self):
        """保存环境回放文件。"""
        self.env.save_replay()

    def get_env_info(self):
        """返回环境的配置信息。"""
        return self.env.get_env_info()

    def get_stats(self):
        """返回环境的统计信息。"""
        return self.env.get_stats()