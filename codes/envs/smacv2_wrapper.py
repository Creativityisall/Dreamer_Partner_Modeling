from pathlib import Path  # 导入用于处理文件路径的 Path 模块
import yaml  # 导入用于读取 YAML 配置文件的库
import numpy as np
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper  # 导入 SMACv2 的环境核心封装器

from .multiagentenv import MultiAgentEnv  # 导入抽象基类 MultiAgentEnv

from typing import List

# 定义 SMACv2 场景配置文件的目录路径，假设配置文件位于当前文件同级目录下的 'smacv2_configs' 文件夹中
SMACv2_CONFIG_DIR = Path(__file__).parent / "smacv2_configs"


def get_scenario_names():
    """
    获取 SMACv2 配置目录中所有场景（YAML 文件）的名称。

    Returns:
        List[str]: 场景名称列表。
    """
    return [p.name for p in SMACv2_CONFIG_DIR.iterdir()]


def load_scenario(map_name, **kwargs):
    """
    加载指定地图名称的 SMACv2 场景配置文件，并用 StarCraftCapabilityEnvWrapper 封装。

    Args:
        map_name (str): 场景的名称（对应 YAML 文件名）。
        **kwargs: 用于覆盖或添加到配置文件的参数。

    Returns:
        StarCraftCapabilityEnvWrapper: 初始化后的 SMACv2 环境实例。
    """
    # 构造 YAML 文件的完整路径
    scenario_path = SMACv2_CONFIG_DIR / f"{map_name}.yaml"

    # 读取 YAML 文件内容
    with open(scenario_path, "r") as f:
        # 使用 FullLoader 加载 YAML 数据
        scenario_args = yaml.load(f, Loader=yaml.FullLoader)

    # 用 kwargs 覆盖或更新配置参数
    scenario_args.update(kwargs)

    # 使用配置中的 env_args 初始化 SMACv2 环境
    return StarCraftCapabilityEnvWrapper(**scenario_args["env_args"])


class SMACv2Wrapper(MultiAgentEnv):
    """
    SMACv2 (StarCraft Capability Environment) 的封装器。
    它加载 YAML 配置文件，并实现了标准的 MultiAgentEnv 接口，
    包括处理回合结束后的吸收状态（Absorbing State）机制。
    """

    def __init__(
            self,
            map_name,
            use_absorbing_state,
            trailing_absorbing_state_length,
            seed,
            **kwargs
    ):
        # 1. 初始化 SMACv2 环境：通过加载 YAML 配置文件来创建环境实例
        self.env = load_scenario(map_name, seed=seed, **kwargs)
        env_info = self.env.get_env_info()

        # 2. 设置环境基本信息
        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]
        self.obs_shape = (env_info["obs_shape"],)
        self.episode_limit = self.env.episode_limit

        # 3. 吸收状态 (Absorbing State) 参数初始化
        self.use_absorbing_state = use_absorbing_state
        self.trailing_absorbing_state_length = trailing_absorbing_state_length  # 吸收状态持续步数

        # 检查参数约束：如果使用吸收状态，则持续步数必须大于 0
        assert not self.use_absorbing_state or self.trailing_absorbing_state_length > 0

        self.is_trailing_absorbing_state = False  # 标志：当前是否处于吸收状态
        self.cur_absorbing_state_length = -1  # 当前已持续的吸收状态步数

        # 定义吸收状态下的观测、动作和奖励等常量
        self.absorbing_obs = np.zeros(self.obs_shape, dtype=np.float32)
        self.absorbing_avail_actions = [0] * self.n_actions
        self.absorbing_avail_actions[0] = 1  # 吸收状态下只允许 NO-OP (动作 0)
        self.absorbing_agent_mask = np.zeros((self.n_agents, 1))  # 智能体掩码为 0
        self.absorbing_rewards = np.zeros((self.n_agents, 1))  # 奖励为 0

    def step(self, actions: List[int]):
        """
        执行环境的一个时间步，包含吸收状态逻辑。

        Args:
            actions (List[int]): 所有智能体的动作列表。

        Returns:
            dict: 包含下一步观测、奖励、终止等信息的字典。
        """
        result = {}

        # --- 1. 正常环境执行阶段 ---
        if not self.is_trailing_absorbing_state:
            # 执行 SMACv2 步骤
            rews, terminated, self.info = self.env.step(actions)

            # 检查是否进入吸收状态：如果环境回合结束且开启了吸收状态
            if self.use_absorbing_state and terminated:
                self.is_trailing_absorbing_state = True
                self.cur_absorbing_state_length = 0
                terminated = False  # 暂时将 terminated 设置为 False，以便在吸收状态中持续执行

            # 获取观测、可用动作和智能体掩码
            obss = self.get_obs()
            avail_actions = self.get_avail_actions()
            agent_mask = self.get_agent_mask()
            # 将奖励形状调整为 (n_agents, 1)，并复制（SMAC通常返回总奖励）
            rewards = np.ones((self.n_agents, 1), dtype=np.float32) * rews

        # --- 2. 吸收状态持续阶段 ---
        else:
            self.cur_absorbing_state_length += 1
            # 检查是否达到吸收状态的持续长度限制，如果达到则真正终止
            terminated = (self.cur_absorbing_state_length >= self.trailing_absorbing_state_length)

            # 使用预定义的吸收状态数据
            obss = [self.absorbing_obs] * self.n_agents
            avail_actions = [self.absorbing_avail_actions] * self.n_agents
            rewards = self.absorbing_rewards
            # 吸收状态下智能体掩码为 0
            agent_mask = self.absorbing_agent_mask

        # --- 3. 构建结果字典 ---
        result = {
            "obs": np.stack(obss, axis=0),  # 形状: (n_agents, obs_dim)
            "avail_actions": np.stack(avail_actions, axis=0),  # 形状: (n_agents, n_actions)
            "agent_mask": agent_mask,  # 形状: (n_agents, 1)
            "rewards": rewards,  # 形状: (n_agents, 1)
            "terminated": np.array([terminated], dtype=np.bool),  # 形状: (1,)
            "truncated": np.array([False], dtype=np.bool),  # 形状: (1,)
            "is_first": np.array([False], dtype=np.bool),  # 形状: (1,)
            "is_trailing_absorbing_state": np.array([self.is_trailing_absorbing_state], dtype=np.bool),  # 形状: (1,)
        }

        # 如果最终回合结束，记录是否获胜
        if terminated:
            result["log_battle_won"] = self.info.get("battle_won", False)

        return result

    def get_obs(self) -> List[np.ndarray]:
        """
        返回所有智能体的当前观测值列表。

        如果智能体已死亡（根据 death_tracker_ally），则返回零填充的吸收观测。
        """
        agents_obs = []
        for i in range(self.n_agents):
            # SMACv2 中 death_tracker_ally[i] == 1 表示智能体 i 已死亡
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

        如果智能体已死亡，则只允许执行 NO-OP (动作 0)。
        """
        avail_actions = []
        for agent_id in range(self.env.n_agents):
            if self.env.death_tracker_ally[agent_id] == 1:
                # 死亡智能体只能执行 NO-OP
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
        # death_tracker_ally: 1=死亡, 0=存活。 1 - tracker: 0=死亡, 1=存活 (即掩码)
        agent_mask = 1 - self.env.death_tracker_ally[..., None]
        return agent_mask

    def reset(self, seed=None, options=None):
        """
        重置环境，并返回初始观测和信息。
        """
        if seed is not None:
            # SMACv2 环境有单独的 seed 方法
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
        self.env.seed(seed)

    def save_replay(self):
        """保存环境回放文件。"""
        self.env.save_replay()

    def get_env_info(self):
        """返回环境的配置信息。"""
        return self.env.get_env_info()

    def get_stats(self):
        """返回环境的统计信息。"""
        return self.env.get_stats()


if __name__ == "__main__":
    # --- 仅用于测试和演示目的的执行块 ---
    for scenario in get_scenario_names():
        env = load_scenario(scenario)
        env_info = env.get_env_info()
        # 打印配置名称、智能体数量、状态形状、观测形状、动作数量
        print(
            scenario,
            env_info["n_agents"],
            env_info["state_shape"],
            env_info["obs_shape"],
            env_info["n_actions"],
        )
        print()