import io
import sys
import traceback
from collections import OrderedDict, defaultdict, deque
from typing import List, Dict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import elements
from elements import UUID
from utils.tools import merge_dict_list, UniformSampler


class Episode:
    """
    Episode 类：用于存储单个回合（轨迹）的所有时间步数据。
    """

    def __init__(self):
        self.time: str = elements.timestamp(millis=True)  # 回合开始时间戳
        self.uuid: UUID = UUID()  # 唯一标识符
        self.data: Dict[str, List[np.ndarray]] = defaultdict(list)  # 存储数据，键为数据类型，值为时间步列表
        self.length: int = 0  # 回合长度（步数）

    @property
    def filename(self):
        """生成用于保存文件的名称：时间-UUID-长度.npz"""
        return f'{self.time}-{str(self.uuid)}-{self.length}.npz'

    def add(self, step_dict: Dict[str, np.ndarray]):
        """将一个时间步的数据添加到回合中。"""
        for key, value in step_dict.items():
            # 忽略以 "log_" 开头的日志数据，只存储环境交互数据
            if not key.startswith("log_"):
                self.data[key].append(value)
        self.length += 1

    def stats(self, rewards_reduce: str) -> Dict[str, float | int]:
        """计算并返回回合的统计数据（例如总奖励、智能体存活率）。"""
        # 计算每个智能体的总奖励
        rewards: np.ndarray = np.array(self.data["rewards"], dtype=np.float32).sum(axis=0).reshape(-1)
        # 计算每个智能体的平均存活掩码（1 - mean(agent_mask) 即死亡率）
        agent_mask: np.ndarray = np.array(self.data["agent_mask"], dtype=np.float32).mean(axis=0).reshape(-1)
        # rewards.shape = (n_agents, )
        metrics = {}

        # 记录每个智能体的指标
        for i in range(rewards.shape[0]):
            metrics[f"agent_{i}/rewards"] = rewards[i]
            metrics[f"agent_{i}/death_ratio"] = 1 - agent_mask[i]

        # 根据配置对奖励进行汇总
        if rewards_reduce == "sum":
            rewards = np.sum(rewards)
        elif rewards_reduce == "mean":
            rewards = np.mean(rewards)
        else:
            raise ValueError(f"Rewards reduce {rewards_reduce} not supported")

        metrics["rewards"] = rewards
        metrics["death_ratio"] = 1 - agent_mask.mean()
        metrics["length"] = len(self) - 1  # 实际交互步数 (通常不计算初始步)
        return metrics

    def save(self, directory) -> Dict:
        """将整个回合数据压缩保存到磁盘 (.npz 文件)。"""
        directory = elements.Path(directory)
        filename = directory / self.filename
        with io.BytesIO() as stream:
            # 使用 np.savez_compressed 压缩存储所有数据
            np.savez_compressed(stream, **self.data)
            stream.seek(0)
            filename.write(stream.read(), mode='wb')
        return {"filename": str(filename)}

    @classmethod
    def load(cls, filename, error="raise"):
        """从磁盘加载一个回合数据。"""
        assert error in ("raise", "none")
        # 从文件名解析时间、UUID 和长度
        time, uuid, length = filename.stem.split("-")[:3]
        length = int(length)
        try:
            with filename.open("rb") as f:
                data = np.load(f)
                data = {k: data[k] for k in data.keys()}  # 转换为字典
        except:
            tb = ''.join(traceback.format_exception(sys.exception()))
            print(f'Error loading chunk {filename}:\n{tb}')
            if error == 'raise':
                raise
            else:
                return None

        episode = cls()
        episode.time = time
        episode.uuid = UUID(uuid)
        episode.data = data
        episode.length = length
        return episode

    def __len__(self) -> int:
        return self.length


# ====================================================================
# Off-Policy Replay Buffer
# ====================================================================

class ReplayBuffer:
    """
    ReplayBuffer 类：用于 Off-Policy 训练（如 Dreamer）的回放缓冲区。
    特点：存储所有已完成的回合，并支持从任意起点采样固定长度的轨迹。
    支持内存溢出到磁盘 (Offload)。
    """

    def __init__(self, config, n_rollout_threads, agg=None):
        # 路径配置
        self.directory = elements.Path(config.logdir) / "replay"
        self.directory.mkdir()

        self.config = config
        # 采样长度：烧机长度 (burn_in) + 批量长度 (batch_length)
        self.length = config.train.burn_in_length + config.train.batch_length
        self.n_rollout_threads = n_rollout_threads
        # 当前正在收集数据的 Episode 列表（按 worker id 索引）
        self.current: List[Episode] = [Episode() for _ in range(n_rollout_threads)]
        # 可用于训练的 Episode 字典 {UUID: Episode}
        self.episodes: Dict[UUID, Episode] = OrderedDict((episode.uuid, episode) for episode in self.current)
        # 均匀采样器，用于高效地从已有的 Episode UUID 中选择一个进行采样
        self.episode_sampler = UniformSampler()
        # 当前缓冲区中所有 Episode 的总步数（用于容量管理）
        self.num_steps_for_training = 0
        # 当 offload 开启时，存储已保存到磁盘的回合的文件路径
        self.uuid_to_filepath: Dict[UUID, str] = dict()
        self._agg = agg  # Episode 统计聚合器

        # 用于磁盘 I/O 的并发线程池，避免阻塞主训练循环
        self.thread_pool = ThreadPoolExecutor(16)
        self.jobs = deque()  # 待执行的保存/加载任务队列

    @elements.timer.section("add")
    def add(self, step_dict: Dict[str, np.ndarray], worker: int):
        """添加一个时间步数据。"""
        episode = self.current[worker]
        episode.add(step_dict)
        self.num_steps_for_training += 1

        # 1. 容量管理：移除最老的 Episode
        with elements.timer.section("remove"):
            if self.config.replay.capacity and self.num_steps_for_training > self.config.replay.capacity:
                oldest_episode_uuid, oldest_episode = next(iter(self.episodes.items()))  # 获取最老的 Episode
                # 更新步数
                self.num_steps_for_training -= (len(oldest_episode) if oldest_episode is not None else 0)
                # 从内存和采样器中删除
                del self.episodes[oldest_episode_uuid]
                del self.episode_sampler[oldest_episode_uuid]
                if self.config.replay.offload:
                    # 如果开启 offload，删除文件路径记录 (但文件本身可能保留，取决于具体实现)
                    del self.uuid_to_filepath[oldest_episode_uuid]

        # 2. 将 Episode 添加到采样器
        # 只要长度大于等于 2，就可以开始采样（因为 Off-Policy 训练需要 prev_actions）
        if episode.uuid not in self.episode_sampler and len(episode) >= 2:
            self.episode_sampler(episode.uuid)

        # 3. 回合结束处理
        done = step_dict["terminated"] or step_dict["truncated"]
        if done:
            # 提交保存任务到线程池
            self.jobs.append(self.thread_pool.submit(episode.save, self.directory))
            if self.config.replay.offload:
                # 如果开启 offload，将 Episode 从内存中卸载（置为 None），并记录文件路径
                filepath = self.directory / episode.filename
                self.uuid_to_filepath[episode.uuid] = str(filepath)
                self.episodes[episode.uuid] = None  # 卸载内存

            # 创建新的 Episode 实例供当前 worker 使用
            new_episode = Episode()
            self.current[worker] = new_episode
            self.episodes[new_episode.uuid] = new_episode

            # 记录已完成 Episode 的统计数据
            if self._agg:
                self._agg.add(episode.stats(rewards_reduce=self.config.logging.rewards_reduce))

    @elements.timer.section("create_dataset")
    def create_dataset(self) -> Dict[str, np.ndarray]:
        """
        生成一个训练批次 (batch)，包含 batch_size 个随机采样的固定长度轨迹。
        """
        # 等待所有正在保存的回合完成（确保数据一致性）
        while self.jobs:
            job = self.jobs.popleft()
            job.result()

        # 并行采样 batch_size 个轨迹
        if self.config.replay.offload:
            samples = [self.thread_pool.submit(self._sample) for _ in range(self.config.train.batch_size)]
            samples: List[Dict[str, np.ndarray]] = [future.result() for future in samples]
        else:
            samples: List[Dict[str, np.ndarray]] = [self._sample() for _ in range(self.config.train.batch_size)]

        # 合并所有样本，通常合并后的形状为 (T, B, ...)
        samples: Dict[str, np.ndarray] = merge_dict_list(samples, axis=1)
        # samples[key].shape = (ts, bs, ...)
        return samples

    def _sample(self) -> Dict[str, np.ndarray]:
        """从一个或多个 Episode 中采样一个固定长度的轨迹。"""
        sample, size = defaultdict(list), 0
        while size < self.length:
            episode_uuid = self.episode_sampler.sample()

            # 从内存加载或从磁盘加载 (如果已卸载)
            episode = self.episodes[episode_uuid]
            if episode is None:
                episode = self.load_episode_from_disk(episode_uuid)

            # 确定采样起点和长度
            # 第一个采样块从随机起点开始，以增加数据多样性
            idx = np.random.randint(len(episode) - 1) if size == 0 else 0
            # 计算可以从当前 Episode 采样的最大长度
            length = min(idx + (self.length - size), len(episode)) - idx

            # 复制数据
            for key in episode.data.keys():
                sample[key].extend(deepcopy(episode.data[key][idx: idx + length]))
            size += length

            # 调整边界条件掩码，确保轨迹是自洽的
            if "is_first" in sample.keys():
                # 采样块的第一个时间步应被标记为 is_first=True
                sample["is_first"][-length] = np.array([True], dtype=bool)
            if "prev_actions" in sample.keys():
                # 采样块的第一个时间步的 prev_actions 设为零（表示边界）
                sample["prev_actions"][-length] = np.zeros_like(sample["prev_actions"][-length])
            if "truncated" in sample.keys():
                # 采样块的最后一个时间步应标记为 truncated=True
                sample["truncated"][-1] = np.array([True], dtype=bool)

        # 组装最终的样本
        for key in sample.keys():
            assert len(sample[key]) == self.length, (key, len(sample[key]))
        sample = {k: np.stack(v) for k, v in sample.items()}
        return sample

    def load_episode_from_disk(self, uuid: UUID):
        """辅助函数：从磁盘加载指定的 Episode。"""
        filepath = elements.Path(self.uuid_to_filepath[uuid])
        episode = Episode.load(filepath)
        return episode

    def clear(self):
        """清空缓冲区，并重新初始化 current Episode。"""
        self.current = [Episode() for _ in range(self.n_rollout_threads)]
        self.episodes = OrderedDict((episode.uuid, episode) for episode in self.current)
        self.episode_sampler = UniformSampler()
        self.num_steps_for_training = 0

    def save(self):
        """保存当前缓冲区状态（此处留空，可能在外部实现）。"""
        pass

    def load(self, data=None, capacity=None):
        """从检查点目录加载先前保存的回合数据到缓冲区。"""
        # ... (加载逻辑，主要用于训练恢复)


# ====================================================================
# On-Policy Replay Buffer
# ====================================================================

class OnPolicyBuffer:
    """
    OnPolicyBuffer 类：用于 On-Policy 训练（如 PPO）的回放缓冲区。
    特点：收集固定长度的轨迹（通常是完整的 rollout），然后训练，随后重置。
    """

    def __init__(self, config, n_rollout_threads, agg=None):
        self.config = config
        self.n_rollout_threads = n_rollout_threads
        # current：当前正在收集的 Episode（当 done 时重置）
        self.current: List[Episode] = [Episode() for _ in range(n_rollout_threads)]
        # completed：用于训练的缓冲区，存储一个完整 epoch 的所有 Episode
        self.completed: List[Episode] = [Episode() for _ in range(n_rollout_threads)]
        self._agg = agg

    @elements.timer.section("add")
    def add(self, step_dict: Dict[str, np.ndarray], worker: int):
        """添加一个时间步数据。"""
        # 数据同时添加到 current（用于统计）和 completed（用于训练）
        self.current[worker].add(step_dict)
        self.completed[worker].add(step_dict)

        # 回合结束处理
        done = step_dict["terminated"] or step_dict["truncated"]
        if done:
            # 记录统计
            if self._agg:
                self._agg.add(self.current[worker].stats(rewards_reduce=self.config.logging.rewards_reduce))
            # 重置 current Episode
            self.current[worker] = Episode()

    @elements.timer.section("create_dataset")
    def create_dataset(self) -> Dict[str, np.ndarray]:
        """
        生成训练批次：将所有 worker 的 completed 轨迹数据进行堆叠。
        通常返回的形状是 (T, B, ...) 或 (B, T, ...)，其中 B 是 n_rollout_threads。
        """
        samples = defaultdict(list)
        for episode in self.completed:
            for key in episode.data.keys():
                # 将一个 Episode 的数据堆叠成 (T, ...) 形状
                sample = np.stack(episode.data[key], axis=0)
                samples[key].append(sample)

        # 将所有 worker 的数据在 Batch 维度 (axis=1) 上堆叠
        samples: Dict[str, np.ndarray] = {k: np.stack(v, axis=1) for k, v in samples.items()}
        # 此时 samples[key].shape = (Time_Step, Batch_Size/n_threads, ...)
        return samples

    def reset(self):
        """
        重置缓冲区：只保留最近一次训练所需的最小数据（通常是 RNN 状态）。
        这个方法在 On-Policy 训练完成后调用。
        """
        # 找到所有 Episode 中数据的最小长度
        min_len = min(len(v) for v in self.completed[0].data.values())
        # 移除已用于训练的数据，只保留 min_len 长度之后的数据（通常是最后一个时间步，用于作为下一个 rollout 的初始 RNN 状态）
        for episode in self.completed:
            for key, value in episode.data.items():
                episode.data[key] = value[min_len:]

    def clear(self):
        """
        彻底清空缓冲区：用于评估模式，因为评估数据不需要保留。
        """
        self.current: List[Episode] = [Episode() for i in range(self.n_rollout_threads)]
        self.completed: List[Episode] = [Episode() for i in range(self.n_rollout_threads)]