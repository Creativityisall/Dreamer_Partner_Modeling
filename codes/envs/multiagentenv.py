# This file is adapted from https://github.com/uoe-agents/epymarl/blob/main/src/envs/multiagentenv.py

class MultiAgentEnv(object):
    """
    多智能体环境抽象基类。
    定义了标准的接口，用于在多智能体强化学习（MARL）框架中封装各种环境（如 StarCraft, MeltingPot, MPE 等）。
    """

    def step(self, actions):
        """
        执行环境的一个时间步。

        Args:
            actions: 所有智能体的动作列表或数组。

        Returns:
            obss (list/array): 所有智能体的下一时刻观测值。
            reward (float/array): 奖励值（可能是总和或每个智能体的奖励）。
            terminated (bool): 是否达到终止状态（D）。
            truncated (bool): 是否达到截断状态（T，如达到最大步数）。
            info (dict): 额外的环境信息。
        """
        raise NotImplementedError

    def get_obs(self):
        """
        返回所有智能体的当前观测值列表或数组。

        Returns:
            list/array: 包含所有智能体观测的集合。
        """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """
        返回指定 ID 智能体的当前观测值。

        Args:
            agent_id (int): 智能体的 ID。

        Returns:
            array: 指定智能体的观测。
        """
        raise NotImplementedError

    def get_obs_size(self):
        """
        返回单个智能体观测值的数据形状/大小。

        Returns:
            int/tuple: 观测值的形状或扁平化后的大小。
        """
        raise NotImplementedError

    def get_state(self):
        """
        返回当前环境的**全局状态**。

        全局状态通常是所有智能体共享的、包含环境完整信息（或足够信息）的表示。
        在中心化训练去中心化执行（CTDE）框架中尤其重要。

        Returns:
            array: 全局状态。
        """
        raise NotImplementedError

    def get_state_size(self):
        """
        返回全局状态的数据形状/大小。

        Returns:
            int/tuple: 全局状态的形状或大小。
        """
        raise NotImplementedError

    def get_avail_actions(self):
        """
        返回所有智能体在当前时间步可用的动作掩码（Avail Actions Mask）。

        用于处理动作空间限制（如 StarCraft 中某些单位无法移动或射击）。

        Returns:
            list/array: 包含所有智能体可用动作掩码的集合。
        """
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """
        返回指定 ID 智能体在当前时间步可用的动作掩码。

        Args:
            agent_id (int): 智能体的 ID。

        Returns:
            array: 指定智能体的可用动作掩码。
        """
        raise NotImplementedError

    def get_total_actions(self):
        """
        返回单个智能体可能采取的**总动作数量**（即动作空间的大小）。

        注意：这通常只适用于每个智能体具有相同离散动作空间的情况。

        Returns:
            int: 动作空间的总大小。
        """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。

        Args:
            seed (int, optional): 随机种子。
            options (dict, optional): 额外的重置选项。

        Returns:
            initial_obss (list/array): 所有智能体的初始观测值。
            info (dict): 初始环境信息。
        """
        raise NotImplementedError

    def render(self):
        """
        渲染环境画面（用于可视化或视频录制）。
        """
        raise NotImplementedError

    def close(self):
        """
        关闭并清理环境资源。
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """
        设置环境的随机种子。
        """
        raise NotImplementedError

    def save_replay(self):
        """
        保存当前回合的回放文件（如果环境支持）。
        """
        raise NotImplementedError

    def get_env_info(self):
        """
        返回包含环境关键信息的字典，通常用于初始化学习算法。

        Returns:
            dict: 包含 state_shape, obs_shape, n_actions, n_agents, episode_limit 的信息。
        """
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,  # 假设 self.n_agents 已在子类中定义
            "episode_limit": self.episode_limit,  # 假设 self.episode_limit 已在子类中定义
        }
        return env_info

    def get_stats(self):
        """
        返回环境的统计信息，如总奖励、步数等，用于日志记录。

        Returns:
            dict: 包含统计信息的字典。
        """
        return {}