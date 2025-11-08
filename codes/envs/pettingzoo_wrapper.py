# The file is adapted from https://github.com/uoe-agents/epymarl/blob/main/src/envs/pz_wrapper.py

import gymnasium as gym
import pettingzoo
import numpy as np

import importlib
from typing import Tuple, List, Dict

from .multiagentenv import MultiAgentEnv  # 假设 MultiAgentEnv 在当前包结构的上层


class PettingZooWrapper(MultiAgentEnv):
    """
    将 PettingZoo (PZ) 风格的环境封装成标准的 MultiAgentEnv 接口。
    PZ 环境使用基于字符串名称的字典进行数据交互，而本封装器将其转换为
    基于智能体索引的 NumPy 数组格式，适用于大多数 MARL 算法框架（如 EPymarl）。
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, lib_name, env_name, seed, **kwargs):
        """
        初始化 PettingZoo 环境。

        Args:
            lib_name (str): PettingZoo 库的名称（如 'mpe', 'atari'）。
            env_name (str): 环境的具体名称（如 'simple_speaker_listener_v4'）。
            seed (int): 环境的随机种子。
            **kwargs: 传递给 PettingZoo 环境的其他参数（如 max_cycles）。
        """
        # 动态导入 PettingZoo 环境模块
        env = importlib.import_module(f"pettingzoo.{lib_name}.{env_name}")
        # 使用 parallel_env 模式，该模式更适合 MARL 训练
        self._env = env.parallel_env(**kwargs)
        # 重置环境并设置种子
        self._env.reset(seed=seed)

        # 智能体数量
        self.n_agents = int(self._env.num_agents)

        # 提取回合长度限制（PZ 的 max_cycles 是回合的步骤数，因此限制应为 max_cycles + 1）
        assert "max_cycles" in kwargs
        self.episode_limit = kwargs["max_cycles"] + 1

        # 提取动作空间和观测空间，注意 PettingZoo 使用列表存储每个智能体的空间
        self.action_space: List[gym.Space] = [self._env.action_space(k) for k in self._env.possible_agents]
        self.observation_space: List[gym.Space] = [self._env.observation_space(k) for k in self._env.possible_agents]

        # 假设所有智能体的动作空间大小和观测形状相同
        self.n_actions = int(self.action_space[0].n)
        self.obs_shape = self.observation_space[0].shape

    def _get_possible_agents_obs(self, obs_dict: Dict[str, np.ndarray]):
        """
        辅助函数：将 PettingZoo 的观测字典（基于智能体名称）转换为 NumPy 数组元组。

        Args:
            obs_dict: PettingZoo step/reset 返回的观测字典。

        Returns:
            tuple: 按照 self._env.possible_agents 顺序排列的观测数组元组。
        """
        obss = []
        for agent in self._env.possible_agents:
            # 确保数据类型为 float32
            obs = np.array(obs_dict[agent], dtype=np.float32)
            obss.append(obs)
        return tuple(obss)

    def reset(self, *args, **kwargs):
        """
        重置环境，并返回初始观测值。
        """
        # 调用 PettingZoo 的 reset
        obs_dict, info_dict = self._env.reset(*args, **kwargs)

        # 将字典观测转换为数组元组
        obss = self._get_possible_agents_obs(obs_dict)

        # 构建符合 MultiAgentEnv 接口的字典结果
        result = {
            # 堆叠观测值，格式: (n_agents, obs_shape...)
            "obs": np.stack(obss, axis=0),
            "agent_mask": self.get_agent_mask(),
            "rewards": np.zeros((self.n_agents, 1), dtype=np.float32),  # 初始奖励为 0
            "terminated": np.array([False], dtype=np.bool),
            "truncated": np.array([False], dtype=np.bool),
            "is_first": np.array([True], dtype=np.bool),
        }
        return result

    def get_agent_mask(self):
        """
        返回一个表示所有智能体都活跃的掩码（全为 1）。
        """
        agent_mask = np.ones((self.n_agents, 1), dtype=np.float32)
        return agent_mask

    def render(self, mode="human"):
        """
        渲染环境画面。
        """
        return self._env.render(mode)

    # TODO: log info
    def step(self, actions: List[int]):
        """
        执行环境的一个时间步。

        Args:
            actions (List[int]): 所有智能体的动作列表，按 self._env.possible_agents 顺序。

        Returns:
            dict: 包含下一步观测、奖励、终止等信息的字典。
        """
        # 1. 动作转换：将动作列表转换为 PettingZoo 需要的字典格式
        dict_actions = {}
        for agent, action in zip(self._env.possible_agents, actions):
            dict_actions[agent] = action

        # 2. 执行 PettingZoo 的 step
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self._env.step(dict_actions)

        # 3. 状态判断：PettingZoo 的 parallel_env 在所有智能体都 terminated/truncated 时才返回 True
        terminated = all([terminated_dict[k] for k in self._env.possible_agents])
        truncated = all([truncated_dict[k] for k in self._env.possible_agents])
        is_first = False

        # 4. 观测转换
        obss = self._get_possible_agents_obs(obs_dict)

        # 5. 奖励处理：计算总奖励（sum），并将其复制给所有智能体（假设是共享奖励）
        # 注意：这里假设是共享奖励（cooperative/team game）。
        rewards_sum = np.array([[reward_dict[k]] for k in self._env.possible_agents], dtype=np.float32).sum()

        # 6. 构建结果字典
        result = {
            "obs": np.stack(obss, axis=0),
            "agent_mask": self.get_agent_mask(),
            # 将总奖励应用于所有智能体
            "rewards": np.ones((self.n_agents, 1), dtype=np.float32) * rewards_sum,
            "terminated": np.array([terminated], dtype=np.bool),
            "truncated": np.array([truncated], dtype=np.bool),
            "is_first": np.array([is_first], dtype=np.bool),
        }
        return result

    # --- MultiAgentEnv 接口实现 ---

    def get_state_size(self) -> Tuple[int]:
        """
        返回全局状态的形状/大小。
        """
        # 依赖于 PettingZoo 环境是否定义了 state_space
        state_shape = self._env.state_space.shape
        return state_shape

    def get_obs_size(self) -> Tuple[int]:
        """
        返回单个智能体观测值的形状/大小。
        """
        obs_shape = self.observation_space[0].shape
        return obs_shape

    def get_total_actions(self) -> List[int]:
        """
        返回所有智能体的总动作数量（每个智能体的动作空间大小）。
        """
        return [action_space.n for action_space in self.action_space]

    def close(self):
        """
        关闭环境。
        """
        return self._env.close()

# 注意：MultiAgentEnv 中其他未实现的抽象方法（如 get_obs, get_state, get_avail_actions 等）
# 在这个 Wrapper 中可能依赖于具体的训练算法需求和 PZ 环境特性，因此没有被实现或被略过。
# 在实际使用中，如果 MARL 算法需要这些方法，需要在 Wrapper 中补充完整。