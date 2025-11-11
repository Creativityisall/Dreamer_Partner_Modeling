from runners.dreamer_runner import DreamerRunner as Runner
from replay import ReplayBuffer
from elements import Agg, FPS
from logger import Logger, TerminalOutput, WandBOutput
from utils import conditions
from world_model import MAWorldModel
from actor_critic.actor import Actor
from actor_critic.critic import Critic
from utils.tools import init_device, get_task_name, make_env, build_returns
from parallel import Remote, Dummy
from partner_model.global_encoder import Global_Encoder
import elements

import numpy as np
import torch
from tensordict.tensordict import TensorDict
torch.set_float32_matmul_precision("high")

from typing import Dict
from copy import deepcopy

class DreamerTrainer:
    def __init__(self, config):
        self.config = config
        self.step = elements.Counter()
        # initialize aggregator
        self.agg = Agg()
        # initialize logger
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
        # initialize envs
        if config.train.parallel_rollout:
            self.envs = [Remote(make_env, config, i) for i in range(config.train.n_rollout_threads)]
        else:
            self.envs = [Dummy(make_env, config, i) for i in range(config.train.n_rollout_threads)]
        # initialize replay
        self.replay = ReplayBuffer(config, config.train.n_rollout_threads, self.agg)
        # initialize device
        self.device = init_device(config)
        # initialize world model
        obs_shape = self.envs[0].obs_shape
        n_actions = self.envs[0].n_actions
        n_agents = self.envs[0].n_agents
        self.wm = MAWorldModel(
            config,
            obs_shape=obs_shape,
            n_actions=n_actions,
            n_agents=n_agents,
            device=self.device,
        )

        # initialize actors
        # 这里的*n_agents表明了这里一定是建立了多个模型
        if config.train.share_actors: # 使用共同的actor
            self.actors = [Actor(config, n_agents, n_actions, 0, self.device)] * n_agents # index默认为0
        else:# 使用独立的actor
            self.actors = [Actor(config, n_agents, n_actions, i, self.device) for i in range(n_agents)]
        # initialize critics
        if config.train.share_critics:
            self.critics = [Critic(config, device=self.device)] * n_agents
        else:
            self.critics = [Critic(config, device=self.device) for _ in range(n_agents)]
        self.target_critics = deepcopy(self.critics)
        # initialize runner
        self.runner = Runner(config, self.envs, self.actors, self.wm, self.replay, self.device)
        # initialize for evaluation
        if config.use_eval:
            if config.eval.parallel_rollout:
                self.eval_envs = [Remote(make_env, config, i) for i in range(config.eval.n_rollout_threads)]
            else:
                self.eval_envs = [Dummy(make_env, config, i) for i in range(config.eval.n_rollout_threads)]
            self.eval_agg = Agg()
            self.eval_replay = ReplayBuffer(config, config.eval.n_rollout_threads, self.eval_agg)
            self.eval_runner = Runner(config, self.eval_envs, self.actors, self.wm, self.eval_replay, self.device)
        # initialize conditions
        batch_steps = self.config.train.batch_size * self.config.train.batch_length
        self.should_train = conditions.Ratio(ratio=(self.config.train.train_ratio / batch_steps), after=self.config.train.prefill_steps, initial=True)
        self.should_eval = conditions.Every(every=config.eval.eval_interval, after=self.config.train.prefill_steps, initial=True)
        self.should_log = conditions.Every(every=self.config.logging.log_interval, after=self.config.train.prefill_steps, initial=True)
        self.should_save = conditions.Every(every=self.config.train.checkpoint.save_interval, after=self.config.train.prefill_steps, initial=True)
        # initialize fps tracker
        self.env_fps = FPS()
        self.train_fps = FPS()
        # setup checkpoint
        self.checkpoint = elements.Checkpoint(directory=config.logdir + "/ckpt", step=self.step)
        self.checkpoint.step = self.step
        self.checkpoint.should_train = self.should_train
        self.checkpoint.should_eval = self.should_eval
        self.checkpoint.should_log = self.should_log
        self.checkpoint.should_save = self.should_save
        for i in range(len(self.actors)):
            setattr(self.checkpoint, f"actor_{i}", self.actors[i])
        for i in range(len(self.critics)):
            setattr(self.checkpoint, f"critic_{i}", self.critics[i])
        for i in range(len(self.target_critics)):
            setattr(self.checkpoint, f"target_critic_{i}", self.target_critics[i])
        self.checkpoint.wm = self.wm
        self.checkpoint.replay = self.replay
        if config.train.checkpoint.from_checkpoint:
            self.checkpoint.load(path=config.train.checkpoint.from_checkpoint)

    def train(self):
        print("Dreamer trainer is running")
        self.runner.reset()
        while self.step < self.config.train.num_env_steps:
            num_steps, _ = self.runner.step(self.agg)
            self.env_fps.step(num_steps)
            self.step.increment(num_steps)

            # train model
            self.train_step()

            # evaluation
            if self.config.use_eval:
                self.eval()

            # log metrics
            self.log_step()

            # save checkpoint
            self.save_step()

        self.close()

    def train_step(self):
        for _ in range(self.should_train(int(self.step))):
            with elements.timer.section("train"):
                self.train_fps.step()

                # train world model
                data: Dict[str, np.ndarray] = self.replay.create_dataset()
                train_metrics = self.wm.update(data)
                self.logger.add(int(self.step), train_metrics, prefix="world_model")

                # generate imaginary transitions
                init_latent = self.wm.prep_init_latent_for_imagination(data)
                terminated = init_latent["terminated"]
                init_latent = init_latent[terminated[:, 0, 0] == 0] if terminated is not None else init_latent
                # init_latent = init_latent
                imaginary_transitions: Dict[str, torch.Tensor] = self.wm.imagine(self.actors, init_latent)
                # imaginary_transitions["deter"].shape = (ts, bs, n_agents, deter_dim)00
                # imaginary_transitions["stoch"].shape = (ts, bs, n_agents, num_classes, stoch_dim)
                # imaginary_transitions["terminated"].shape = (ts, bs, n_agents, 1)
                # imaginary_transitions["rewards"].shape = (ts, bs, n_agents, 1)
                # imaginary_transitions["actions_env"].shape = (ts-1, bs, n_agents)
                # imaginary_transitions["avail_actions"].shape = (ts, bs, n_agents, n_actions)

                latent = TensorDict(
                    {
                        "deter": imaginary_transitions["deter"].detach(),
                        "stoch": imaginary_transitions["stoch"].detach(),
                    },
                    batch_size=imaginary_transitions["deter"].shape[:-1],
                )

                # calculate the value targets
                value_preds_list = []
                for i in range(self.runner.n_agents):
                    value_preds = self.target_critics[i](latent[:, :, i])["value_preds"]
                    value_preds_list.append(value_preds)
                value_preds = torch.stack(value_preds_list, dim=2)
                rewards = imaginary_transitions["rewards"]
                rewards = rewards.mean(dim=2, keepdim=True).expand_as(rewards)
                terminated = imaginary_transitions["terminated"].detach()
                target_returns = build_returns(
                    rewards=rewards,
                    value_preds=value_preds,
                    terminated=terminated,
                    truncated=torch.zeros_like(terminated),
                    gamma=self.config.train.gamma,
                    gae_lambda=self.config.train.gae_lambda,
                )

                # calculate the advantages
                advantages = []
                for i in range(len(self.actors)):
                    advantage = target_returns[:, :, i] - value_preds[:, :, i]
                    # 对advantage进行归一化
                    advantage_mean = advantage.mean()
                    advantage_std = advantage.std()
                    advantage = (advantage - advantage_mean) / (advantage_std + 1e-5)
                    advantages.append(advantage)
                advantages = torch.stack(advantages, dim=2).detach()

                self.logger.add(
                    int(self.step),
                    {
                        "rewards": imaginary_transitions["rewards"].mean().item(),
                        "terminated": imaginary_transitions["terminated"].mean().item(),
                        "target_returns": target_returns.mean().item(),
                        "value_preds": value_preds.mean().item(),
                    },
                    prefix="imagination",
                )

                # train actor
                if self.config.train.share_actors:
                    train_metrics = self.actors[0].ppo_update(
                        latent=latent[:-1],
                        advantages=advantages[:-1],
                        actions_env=imaginary_transitions["actions_env"],
                        avail_actions=imaginary_transitions["avail_actions"][:-1] if "avail_actions" in imaginary_transitions else None,
                        global_vectors=imaginary_transitions["global_vectors"],
                        local_vectors=imaginary_transitions["local_vectors"],
                    )
                    self.logger.add(int(self.step), train_metrics, prefix="agent_0")
                else:
                    # TODO:从这里加入local变量对齐
                    for i in range(len(self.actors)):
                        train_metrics = self.actors[i].ppo_update(
                            latent=latent[:-1, :, i],
                            advantages=advantages[:-1, :, i],
                            actions_env=imaginary_transitions["actions_env"][:, :, i],
                            avail_actions=imaginary_transitions["avail_actions"][:-1, :, i] if "avail_actions" in imaginary_transitions else None,
                            global_vectors=imaginary_transitions["global_vectors"],
                            local_vectors=imaginary_transitions["local_vectors"],
                        )
                        self.logger.add(int(self.step), train_metrics, prefix=f"agent_{i}")

                # train critic
                if self.config.train.share_critics:
                    train_metrics = self.critics[0].ppo_update(
                        latent=latent[:-1],
                        target_returns=target_returns[:-1],
                    )
                    self.logger.add(int(self.step), train_metrics, prefix="agent_0")
                else:
                    for i in range(len(self.critics)):
                        train_metrics = self.critics[i].ppo_update(
                            latent=latent[:-1, :, i],
                            target_returns=target_returns[:-1, :, i],
                        )
                        self.logger.add(int(self.step), train_metrics, prefix=f"agent_{i}")

                # update target critic
                if self.config.train.share_critics:
                    for param, target_param in zip(self.critics[0].parameters(), self.target_critics[0].parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - self.config.train.target_update_tau)
                            + param.data * self.config.train.target_update_tau
                        )
                else:
                    for i in range(len(self.critics)):
                        for param, target_param in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                            target_param.data.copy_(
                                target_param.data * (1.0 - self.config.train.target_update_tau)
                                + param.data * self.config.train.target_update_tau
                            )
                # update global_encoder
                # TODO:注意全局的encoder应该在这里外部更新
                self.wm.global_encoder.update(imaginary_transitions["global_vectors"], imaginary_transitions["local_vectors"])


    @torch.no_grad()
    def eval(self):
        if self.should_eval(int(self.step)):
            with elements.timer.section("eval"):
                self.eval_replay.clear()
                self.eval_runner.reset()
                episodes = elements.Counter()
                while episodes < self.config.eval.eval_episode_num:
                    _, num_episodes = self.eval_runner.step(self.eval_agg, evaluation=True)
                    episodes.increment(num_episodes)
                self.logger.add(int(self.step), self.eval_agg.result(reset=True, prefix="eval"))

            # # video preds
            # data: Dict[str, np.ndarray] = self.eval_replay.create_dataset()
            # video, rewards_preds = self.wm.video_preds(data)
            # video = video[:, 0, 0]
            # imageio.mimsave("video_preds.gif", video, fps=5)
            # # assert False

            # # save episode
            # sample, episode_uuid = self.eval_replay.sample_one_episode()
            # # obs = sample["obs"][:, 0].astype(np.uint8)
            # state = sample["state"][:].astype(np.uint8)
            # rewards = sample["rewards"][:].astype(np.float32)

            # # imageio.mimsave("video_preds.gif", obs, fps=5)
            # imageio.mimsave(f"state_preds_{episode_uuid}_{rewards.sum()}.gif", state, fps=5)
            # assert False, rewards_preds[:, 0].sum()

    def log_step(self):
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
        with elements.timer.section("save"):
            if self.should_save(int(self.step)):
                self.checkpoint.save()

    def close(self):
        for env in self.envs:
            env.close()
        self.logger.close()
