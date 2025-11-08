# 导入所需的库
from meltingpot import substrate  # 导入MeltingPot环境基类
import dm_env  # DeepMind Environment 接口，MeltingPot基于此
from gymnasium import spaces  # Gym/Gymnasium 空间定义库
import numpy as np  # 数值计算
import tree  # DeepMind的tree库，用于处理嵌套数据结构（如spec）
import cv2  # OpenCV，用于图像处理，这里用于可选的观测值大小调整

from typing import List, Tuple, Dict  # 类型提示

from envs.multiagentenv import MultiAgentEnv  # 假设的自定义多智能体环境基类


def timestep_to_RGB_obs(timestep: dm_env.TimeStep, reshape: Tuple[int, int] = None) -> List[np.ndarray]:
    """
    将 dm_env.TimeStep 对象的观测值（observation）转换为 RGB 图像的 NumPy 数组列表。

    MeltingPot 的观测值是结构化的（Dict），我们只提取其中的 "RGB" 键。

    Args:
        timestep: 当前时间步的 dm_env.TimeStep 对象。
        reshape: 可选的 (宽度, 高度) 元组，用于调整 RGB 图像的大小。

    Returns:
        包含所有智能体 RGB 观测值的 NumPy 数组列表。
    """
    # 提取所有智能体的 RGB 观测值，并转换为 float32 类型
    obss = [np.array(obs["RGB"], dtype=np.float32) for obs in timestep.observation]

    # 如果指定了 reshape 大小，则使用 OpenCV 进行调整
    if reshape:
        # cv2.resize 使用 INTER_AREA 进行插值，适用于图像缩减
        obss = [cv2.resize(obs, reshape, interpolation=cv2.INTER_AREA) for obs in obss]

    # 将所有智能体的观测值堆叠成一个 NumPy 数组 (num_agents, H, W, C)
    obss = np.stack(obss, axis=0)
    return obss


def timestep_to_avail_actions(timestep: dm_env.TimeStep) -> List[np.ndarray]:
    """
    根据智能体的状态生成可用的动作掩码（Avail Actions Mask）。

    MeltingPot 中的 'READY_TO_SHOOT' 键决定了第 7 个动作（SHOOT）是否可用。

    Args:
        timestep: 当前时间步的 dm_env.TimeStep 对象。

    Returns:
        包含所有智能体可用动作掩码的 NumPy 数组 (num_agents, num_actions)。
    """
    # 提取所有智能体的 READY_TO_SHOOT 状态（布尔值）
    ready_to_shoot = np.array([obs["READY_TO_SHOOT"] for obs in timestep.observation]).astype(np.bool)

    # 初始化一个所有动作都可用的掩码 (8 个动作)
    avail_actions = np.ones((len(ready_to_shoot), 8))

    # 如果智能体不处于 READY_TO_SHOOT 状态 (~ready_to_shoot)，则禁用第 7 个动作（SHOOT）
    avail_actions[~ready_to_shoot, 7] = 0
    return avail_actions


def remove_world_observations_from_space(observation: spaces.Dict) -> spaces.Dict:
    """
    从观测空间中移除世界级（WORLD.）的观测，只保留智能体私有观测。
    """
    return spaces.Dict({
        key: observation[key] for key in observation if "WORLD." not in key
    })


def spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
    """
    将 dm_env 的嵌套 Spec 结构（用于定义数据形状和类型）转换为 Gymnasium/Gym 的 Space 结构。

    Args:
        spec: dm_env.specs.Array 的嵌套结构。

    Returns:
        对应的 Gymnasium/Gym Space。
    """
    if isinstance(spec, dm_env.specs.DiscreteArray):
        # 离散数组转换为 Discrete 空间
        return spaces.Discrete(spec.num_values)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        # 有界数组转换为 Box 空间
        return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        # 无界数组转换为 Box 空间
        if np.issubdtype(spec.dtype, np.floating):
            return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
        elif np.issubdtype(spec.dtype, np.integer):
            info = np.iinfo(spec.dtype)
            return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
        else:
            raise NotImplementedError(f'Unsupported dtype {spec.dtype}')
    elif isinstance(spec, (list, tuple)):
        # 列表或元组递归转换为 Tuple 空间
        return spaces.Tuple([spec_to_space(s) for s in spec])
    elif isinstance(spec, dict):
        # 字典递归转换为 Dict 空间
        return spaces.Dict({key: spec_to_space(s) for key, s in spec.items()})
    else:
        raise ValueError('Unexpected spec of type {}: {}'.format(type(spec), spec))


class MeltingPotWrapper(MultiAgentEnv):
    """
    MeltingPot 环境的封装器，用于将其适配到标准的 MultiAgentEnv 接口。
    """

    def __init__(self, env_config: Dict):
        # 使用配置中的 substrate 名称和角色列表构建 MeltingPot 环境
        self._env = substrate.build(env_config['substrate'], roles=env_config['roles'])

        # 获取智能体数量
        self.n_agents = len(self._env.observation_spec())

        # 转换第一个智能体的观测 Spec 为 Gym Space，并移除世界观测
        self.observation_space = remove_world_observations_from_space(spec_to_space(self._env.observation_spec()[0]))

        # 转换第一个智能体的动作 Spec 为 Gym Space
        self.action_space = spec_to_space(self._env.action_spec()[0])

        # 获取配置中的图像大小调整参数
        self.reshape = env_config.get("reshape", None)

        # 计算经过调整后的观测形状 (H, W, C)
        self.obs_shape = (*self.reshape, 3) if self.reshape else self.observation_space["RGB"].shape

        # 获取动作空间大小
        self.n_actions = self.action_space.n.item()

        # 设置最大时间步数（用于判断截断 T）
        self.max_cycles = env_config["max_cycles"]

        # 是否在结果中包含可用动作掩码
        self.use_avail_actions = env_config.get("use_avail_actions", False)

    def reset(self, *args, **kwargs):
        """
        重置环境，并返回初始观测值。
        """
        # 调用 MeltingPot 环境的 reset 方法
        timestep = self._env.reset()
        self.num_steps = 0  # 重置步数计数

        # 转换 RGB 观测值
        obs = timestep_to_RGB_obs(timestep, self.reshape)

        # 提取 READY_TO_SHOOT 状态（注意：这里使用 timestep.observation 而非 obs，因为 obs 仅包含 RGB）
        # 修复：这里的 READY_TO_SHOOT 提取应基于原始 timestep.observation
        ready_to_shoot = np.array([o["READY_TO_SHOOT"] for o in timestep.observation])

        # 生成可用动作掩码
        avail_actions = timestep_to_avail_actions(timestep)

        # 构建符合 MultiAgentEnv 接口的字典结果
        result = {
            "obs": obs,  # 格式: (num_agents, H, W, C)
            "READY_TO_SHOOT": ready_to_shoot,
            # "state": state, # 状态（世界 RGB），已注释掉
            "agent_mask": self.get_agent_mask(),  # 智能体掩码（全为 1，表示所有智能体都活跃）
            "rewards": np.zeros((self.n_agents, 1), dtype=np.float32),  # 初始奖励为 0
            "terminated": np.array([False], dtype=np.bool),  # 初始未终止
            "truncated": np.array([False], dtype=np.bool),  # 初始未截断
            "is_first": np.array([True], dtype=np.bool),  # 表示这是第一个时间步
        }

        # 如果启用，添加可用动作掩码
        if self.use_avail_actions:
            result["avail_actions"] = avail_actions

        return result

    def step(self, actions: List[int]):
        """
        执行一步动作，并返回下一个状态、奖励等信息。

        Args:
            actions: 所有智能体的动作列表。

        Returns:
            包含下一步信息的字典。
        """
        # 调用 MeltingPot 环境的 step 方法
        timestep = self._env.step(actions)
        self.num_steps += 1  # 增加步数计数

        # 转换 RGB 观测值
        obs = timestep_to_RGB_obs(timestep, self.reshape)

        # 提取 READY_TO_SHOOT 状态
        ready_to_shoot = np.array([o["READY_TO_SHOOT"] for o in timestep.observation])

        # 生成可用动作掩码
        avail_actions = timestep_to_avail_actions(timestep)

        # state = self.render().astype(np.float32) # 世界状态，已注释掉

        # 提取奖励，并调整形状为 (num_agents, 1)
        rewards = [[timestep.reward[index]] for index in range(self.n_agents)]

        # 检查是否达到最大步数（截断 T）
        truncated = self.num_steps >= self.max_cycles

        # 检查环境是否结束（终止 D）
        terminated = timestep.last()

        # 构建符合 MultiAgentEnv 接口的字典结果
        result = {
            "obs": obs,
            # "state": state,
            "READY_TO_SHOOT": ready_to_shoot,
            "agent_mask": self.get_agent_mask(),
            "rewards": np.array(rewards, dtype=np.float32),
            "terminated": np.array([terminated], dtype=np.bool),
            "truncated": np.array([truncated], dtype=np.bool),
            "is_first": np.array([False], dtype=np.bool),
        }

        # 如果启用，添加可用动作掩码
        if self.use_avail_actions:
            result["avail_actions"] = avail_actions

        return result

    def get_avail_actions(self):
        """
        此方法在 step 中实现，这里留空。
        """
        pass

    def close(self):
        """
        关闭 MeltingPot 环境。
        """
        self._env.close()

    def get_agent_mask(self):
        """
        返回一个表示所有智能体都活跃的掩码 (全为 1)。
        """
        agent_mask = np.ones((self.n_agents, 1), dtype=np.float32)
        return agent_mask

    def render(self) -> np.ndarray:
        """
        渲染环境，返回世界级的 RGB 图像。

        用于录制视频或可视化。

        Returns:
            np.ndarray: 世界的 RGB 图像 (H, W, 3)。
        """
        # 获取当前时间步的完整观测
        observation = self._env.observation()
        # 提取第一个智能体的世界 RGB 观测（假设所有智能体看到的世界 RGB 都一样）
        world_rgb = observation[0]['WORLD.RGB']

        # 返回世界 RGB 图像
        return world_rgb