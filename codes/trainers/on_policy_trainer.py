from runners.on_policy_runner import OnPolicyRunner as Runner
from replay import OnPolicyBuffer
from elements import Agg, FPS
from logger import Logger, TerminalOutput, WandBOutput
from actor_critic.rnn_actor import RNNActor  # 导入基于 RNN 的策略网络
from actor_critic.rnn_critic import RNNCritic  # 导入基于 RNN 的价值网络
from utils import conditions
from utils.tools import init_device, get_task_name, make_env, n2t, build_returns
from parallel import Remote, Dummy
import elements

import torch

torch.set_float32_matmul_precision("high")


class OnPolicyTrainer:
    """
    OnPolicyTrainer 类：实现基于 RNN 的 On-Policy (例如 PPO) 多智能体强化学习的训练逻辑。
    它使用 OnPolicyBuffer 收集完整轨迹，然后进行批量训练。
    """

    def __init__(self, config):
        self.config = config
        self.step = elements.Counter()  # 环境总步数计数器

        # --------------------------- 初始化日志和配置 ---------------------------
        self.agg = Agg()  # 数据聚合器
        # 初始化 Logger
        output_handles = [TerminalOutput(pattern=config.logging.terminal_filter)]
        if config.logging.use_wandb:
            output_handles.append(
                WandBOutput(
                    name=config.logdir.split("/")[-1],
                    pattern=config.logging.wandb_filter,
                    config=config,
                    group=config.env + "." + get_task_name(config),
                    **config.logging.wandb_config,
                )
            )
        self.logger = Logger(output_handles=output_handles)

        # --------------------------- 初始化环境和数据 ---------------------------
        # 初始化训练环境（并行环境）
        if config.train.parallel_rollout:
            self.envs = [Remote(make_env, config, i) for i in range(config.train.n_rollout_threads)]
        else:
            self.envs = [Dummy(make_env, config, i) for i in range(config.train.n_rollout_threads)]

        # 初始化 On-Policy 回放缓冲区 (OnPolicyBuffer)，用于存储最近一次 rollout 的数据
        self.replay = OnPolicyBuffer(config, config.train.n_rollout_threads, self.agg)

        # --------------------------- 初始化设备和模型 ---------------------------
        self.device = init_device(config)
        # 定义类型和设备字典 (tpdv)，方便将 numpy 数组转为 Tensor
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        # 获取环境信息用于模型初始化
        obs_shape = self.envs[0].obs_shape
        n_actions = self.envs[0].n_actions
        n_agents = self.envs[0].n_agents

        # 初始化 Actor (RNNActor)
        if config.train.share_actors:
            self.actors = [RNNActor(config, obs_shape, n_agents, n_actions, self.device)] * n_agents
        else:
            self.actors = [RNNActor(config, obs_shape, n_agents, n_actions, self.device) for _ in range(n_agents)]

        # 初始化 Critic (RNNCritic)
        if config.train.share_critics:
            self.critics = [RNNCritic(config, obs_shape, self.device)] * n_agents
        else:
            self.critics = [RNNCritic(config, obs_shape, self.device) for _ in range(n_agents)]

        # 初始化 Runner，负责环境交互和数据收集
        self.runner = Runner(
            config=config,
            envs=self.envs,
            actors=self.actors,
            critics=self.critics,
            replay=self.replay,
            device=self.device,
        )

        # --------------------------- 初始化评估环境 ---------------------------
        if config.use_eval:
            if config.eval.parallel_rollout:
                self.eval_envs = [Remote(make_env, config, i) for i in range(config.eval.n_rollout_threads)]
            else:
                self.eval_envs = [Dummy(make_env, config, i) for i in range(config.eval.n_rollout_threads)]
            self.eval_agg = Agg()
            self.eval_replay = OnPolicyBuffer(config, config.eval.n_rollout_threads, self.eval_agg)
            self.eval_runner = Runner(
                config=config,
                envs=self.eval_envs,
                actors=self.actors,
                critics=self.critics,
                replay=self.eval_replay,
                device=self.device,
            )

        # --------------------------- 初始化训练条件 ---------------------------
        # 训练频率：每当收集够 train_every 步数据（n_rollout_threads * batch_length）后训练一次
        train_every = config.train.n_rollout_threads * config.train.batch_length
        self.should_train = conditions.Every(every=train_every, initial=False)
        self.should_eval = conditions.Every(every=config.eval.eval_interval, initial=False)
        self.should_log = conditions.Every(every=config.logging.log_interval, initial=False)
        self.should_save = conditions.Every(every=self.config.train.checkpoint.save_interval, initial=False)

        # --------------------------- 初始化 FPS 和检查点 ---------------------------
        self.env_fps = FPS()
        self.train_fps = FPS()

        # 设置 Checkpoint 对象，用于保存和加载模型/状态
        self.checkpoint = elements.Checkpoint(directory=config.logdir + "/ckpt", step=self.step)
        self.checkpoint.step = self.step
        self.checkpoint.should_train = self.should_train
        self.checkpoint.should_eval = self.should_eval
        self.checkpoint.should_log = self.should_log
        self.checkpoint.should_save = self.should_save
        # 注册 Actor 和 Critic
        for i in range(len(self.actors)):
            setattr(self.checkpoint, f"actor_{i}", self.actors[i])
        for i in range(len(self.critics)):
            setattr(self.checkpoint, f"critic_{i}", self.critics[i])

        # 从检查点加载
        if config.train.checkpoint.from_checkpoint:
            self.checkpoint.load(path=config.train.checkpoint.from_checkpoint)

    def train(self):
        """主训练循环"""
        print("On-policy trainer is running")
        self.runner.reset()
        while self.step < self.config.train.num_env_steps:
            # 1. 环境交互：收集一个 batch_length 长度的轨迹
            num_steps, _ = self.runner.step(self.agg)
            self.env_fps.step(num_steps)
            self.step.increment(num_steps)

            # 2. 模型训练：达到训练条件后执行
            self.train_step()

            # 3. 评估
            if self.config.use_eval:
                self.eval()

            # 4. 记录日志
            self.log_step()

            # 5. 保存检查点
            self.save_step()

        self.close()

    @elements.timer.section("train")
    def train_step(self):
        """执行模型的 PPO 训练步骤"""
        if self.should_train(int(self.step)):
            self.train_fps.step()

            # ------------------------- 准备数据 -------------------------
            # 从 OnPolicyBuffer 采样整个 rollout 轨迹
            data = self.replay.create_dataset()
            # 将 numpy 数据转为 Tensor，并移到设备上 (n2t: numpy to tensor)
            obs = n2t(data["obs"], **self.tpdv)
            rnn_states = n2t(data["rnn_states"], **self.tpdv)  # Actor 的初始 RNN 状态
            rnn_states_critic = n2t(data["rnn_states_critic"], **self.tpdv)  # Critic 的初始 RNN 状态
            value_preds = n2t(data["value_preds"], **self.tpdv)  # 之前 roll-out 时 Critic 的预测值
            rewards = n2t(data["rewards"], **self.tpdv)
            terminated = n2t(data["terminated"], **self.tpdv)
            truncated = n2t(data["truncated"], **self.tpdv)
            agent_mask = n2t(data["agent_mask"], **self.tpdv)  # 智能体存活/参与掩码 (用于 MARL)
            actions_env = n2t(data["actions_env"], **self.tpdv)
            avail_actions = n2t(data["avail_actions"], **self.tpdv) if "avail_actions" in data else None

            # ------------------------- 计算目标回报 (Target Returns) -------------------------
            # 1. 预测最后一个时间步的价值（用于回报计算的边界条件）
            value_preds_list = []
            for i in range(self.runner.n_agents):
                # 使用 Critic 预测轨迹最后一个时间步的 V-pred
                # obs[-1:, :, i] 是最后一个时间步的观测，rnn_states_critic[-1:, :, i] 是相应的 RNN 状态
                last_value_preds = self.critics[i](obs[-1:, :, i], rnn_states_critic[-1:, :, i])["value_preds"]
                # 将 rollout 轨迹上的 V-preds 和最后一个时间步的预测值拼接
                value_preds_list.append(torch.cat([value_preds[:, :, i], last_value_preds], dim=0))
            value_preds = torch.stack(value_preds_list, dim=2)

            # 2. 共享奖励处理：通常在协作 MARL 中，奖励是共享的
            rewards = rewards.mean(dim=2, keepdim=True).expand_as(rewards)  # 取所有智能体奖励的平均值作为共享奖励

            # 3. 使用 GAE (Generalized Advantage Estimation) 计算 Target Returns
            target_returns = build_returns(
                rewards=rewards,
                value_preds=value_preds,
                terminated=terminated.unsqueeze(2),
                truncated=truncated.unsqueeze(2),
                gamma=self.config.train.gamma,
                gae_lambda=self.config.train.gae_lambda,
            )

            # ------------------------- 计算优势函数 (Advantages) -------------------------
            advantages = []
            for i in range(len(self.actors)):
                # GAE优势 = Target Return - V-pred
                advantage = target_returns[:, :, i] - value_preds[:, :, i]
                # 对优势进行标准化（只对 agent_mask 为 1 的有效数据进行均值和标准差计算）
                advantage_mean = advantage[agent_mask[:, :, i] == 1].mean()
                advantage_std = advantage[agent_mask[:, :, i] == 1].std()
                advantages.append((advantage - advantage_mean) / (advantage_std + 1e-5))
            advantages = torch.stack(advantages, dim=2)

            # ------------------------- 训练 Actor -------------------------
            if self.config.train.share_actors:
                # 共享策略：使用所有数据更新一个 Actor
                train_metrics = self.actors[0].ppo_update(
                    obs=obs[:-1],  # 排除最后一个时间步的观测
                    rnn_states=rnn_states[:-1],  # 排除最后一个时间步的 RNN 状态
                    actions_env=actions_env,
                    agent_mask=agent_mask[:-1],
                    advantages=advantages[:-1],
                    avail_actions=avail_actions[:-1] if avail_actions is not None else None,
                )
                self.logger.add(int(self.step), train_metrics, prefix="agent_0")
            else:
                # 独立策略：迭代更新每个智能体的 Actor
                for i in range(len(self.actors)):
                    train_metrics = self.actors[i].ppo_update(
                        obs=obs[:-1, :, i],# obs: 时间步, 维度, agent_index
                        rnn_states=rnn_states[:-1, :, i],
                        actions_env=actions_env[:, :, i],
                        agent_mask=agent_mask[:-1, :, i],
                        advantages=advantages[:-1, :, i],
                        avail_actions=avail_actions[:-1, :, i] if avail_actions is not None else None,
                    )
                    self.logger.add(int(self.step), train_metrics, prefix=f"agent_{i}")

            # ------------------------- 训练 Critic -------------------------
            if self.config.train.share_critics:
                # 共享价值：使用所有数据更新一个 Critic
                train_metrics = self.critics[0].ppo_update(
                    obs=obs[:-1],
                    rnn_states_critic=rnn_states_critic[:-1],
                    target_returns=target_returns[:-1],
                    agent_mask=agent_mask[:-1],
                )
                self.logger.add(int(self.step), train_metrics, prefix="agent_0")
            else:
                # 独立价值：迭代更新每个智能体的 Critic
                for i in range(len(self.critics)):
                    train_metrics = self.critics[i].ppo_update(
                        obs=obs[:-1, :, i],
                        rnn_states_critic=rnn_states_critic[:-1, :, i],
                        target_returns=target_returns[:-1, :, i],
                        agent_mask=agent_mask[:-1, :, i],
                    )
                    self.logger.add(int(self.step), train_metrics, prefix=f"agent_{i}")

            # 4. 重置回放缓冲区 (On-Policy 训练后数据即被丢弃)
            self.replay.reset()

    @torch.no_grad()
    def eval(self):
        """评估步骤"""
        if self.should_eval(int(self.step)):
            with elements.timer.section("eval"):
                self.eval_replay.clear()
                self.eval_runner.reset()
                episodes = elements.Counter()
                # 运行评估回合，直到达到设定的数量
                while episodes < self.config.eval.eval_episode_num:
                    _, num_episodes = self.eval_runner.step(self.eval_agg, evaluation=True)
                    episodes.increment(num_episodes)
                # 记录评估结果
                self.logger.add(int(self.step), self.eval_agg.result(reset=True, prefix="eval"))

    def log_step(self):
        """记录日志步骤"""
        if self.should_log(int(self.step)):
            with elements.timer.section("log"):
                self.logger.add(int(self.step), self.agg.result(reset=True))
                self.logger.add(
                    int(self.step),
                    {
                        "env_fps": self.env_fps.result(reset=False),
                        "train_fps": self.train_fps.result(reset=False),
                    }
                )
                if self.config.logging.timer:
                    timer_dict = elements.timer.stats()
                    timer_dict.pop('summary')
                    self.logger.add(int(self.step), timer_dict, prefix="timer")
                self.logger.flush()

    def save_step(self):
        """保存检查点步骤"""
        if self.should_save(int(self.step)):
            with elements.timer.section("save"):
                self.checkpoint.save()

    def close(self):
        """关闭所有环境和 Logger"""
        for env in self.envs:
            env.close()
        self.logger.close()