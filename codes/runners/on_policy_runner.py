import torch
from tensordict.tensordict import TensorDict
import numpy as np

from typing import Dict, List
import re

class OnPolicyRunner:
    def __init__(self, config, envs, actors, critics, replay, device=torch.device("cpu")):
        self.config = config
        self.envs = envs
        self.actors = actors
        self.critics = critics
        self.replay = replay
        self.n_agents = len(actors)
        self.device = device

    def reset(self):
        # initialize the environments
        futures = [env.reset() for env in self.envs]
        self.step_dict_list: List[Dict[str, np.ndarray]] = [future() for future in futures]
        self.merged_step_dict: TensorDict = torch.stack(
            [
                TensorDict(
                    step_dict,
                    device=self.device,
                ).named_apply(lambda k, v: v if not k.startswith("log_") else None) for step_dict in self.step_dict_list
            ],
        )

        # initialize the agent state
        self.agent_state: TensorDict = self.initialize_agent_state(batch_size=len(self.envs))

        # store the first step
        for i, step_dict in enumerate(self.step_dict_list):
            step_dict["rnn_states"] = self.agent_state["rnn_states"][i].cpu().numpy()
            step_dict["rnn_states_critic"] = self.agent_state["rnn_states_critic"][i].cpu().numpy()
            self.replay.add(step_dict, worker=i)

    def step(self, agg=None, evaluation=False):
        actor_outputs, critic_outputs, self.agent_state = self.get_actions(
            obs=self.merged_step_dict["obs"],
            agent_state=self.agent_state,
            avail_actions=(self.merged_step_dict["avail_actions"] if "avail_actions" in self.merged_step_dict else None),
            evaluation=evaluation,
        )
        dones = self.merged_step_dict["terminated"] | self.merged_step_dict["truncated"]
        actions_env = actor_outputs["actions_env"].cpu().numpy()
        futures = [
            env.step(actions_env[i]) if not dones[i]
            else env.reset()
            for i, env in enumerate(self.envs)
        ]
        self.step_dict_list: List[Dict[str, np.ndarray]] = [future() for future in futures]
        self.merged_step_dict: TensorDict = torch.stack(
            [
                TensorDict(
                    step_dict,
                    device=self.device,
                ).named_apply(lambda k, v: v if not k.startswith("log_") else None) for step_dict in self.step_dict_list
            ],
        )

        # reset agent state after env reset
        is_first = self.merged_step_dict["is_first"].squeeze(-1)
        if is_first.any():
            self.agent_state[is_first] = self.initialize_agent_state(batch_size=is_first.sum().item())

        # add step_dict to replay
        for i, step_dict in enumerate(self.step_dict_list):
            step_dict["rnn_states"] = self.agent_state["rnn_states"][i].cpu().numpy()
            step_dict["rnn_states_critic"] = self.agent_state["rnn_states_critic"][i].cpu().numpy()
            step_dict["actions_env"] = actor_outputs["actions_env"][i].cpu().numpy()
            step_dict["value_preds"] = critic_outputs["value_preds"][i].cpu().numpy()
            self.replay.add(step_dict, worker=i)

        # aggregate env stats
        if agg is not None:
            for step_dict in self.step_dict_list:
                for key, value in step_dict.items():
                    if key.startswith("log_"):
                        if re.match(self.config.logging.log_keys_avg, key):
                            agg.add(key, value, agg="avg")
                        if re.match(self.config.logging.log_keys_sum, key):
                            agg.add(key, value, agg="sum")
                        if re.match(self.config.logging.log_keys_max, key):
                            agg.add(key, value, agg="max")

        # calculate the number of steps and episodes
        num_steps = 0
        num_episodes = 0
        for step_dict in self.step_dict_list:
            done = step_dict["terminated"] or step_dict["truncated"]
            num_episodes += done
            if self.config.env_args.get("use_absorbing_state", False):
                num_steps += not step_dict["is_trailing_absorbing_state"]
            else:
                num_steps += not step_dict["is_first"]

        return num_steps, num_episodes

    def get_actions(
            self,
            obs: torch.Tensor,
            agent_state: TensorDict,
            avail_actions: torch.Tensor | None = None,
            evaluation: bool = False,
    ):
        critic_outputs: List[TensorDict] = [self.critics[i](
            obs=obs[:, i],
            rnn_states_critic=agent_state["rnn_states_critic"][:, i],
        ) for i in range(len(self.critics))]
        critic_outputs = torch.stack(critic_outputs, dim=1)
        actor_outputs: List[TensorDict] = [
            self.actors[i](
                obs=obs[:, i],
                rnn_states=agent_state["rnn_states"][:, i],
                avail_actions=avail_actions[:, i] if avail_actions is not None else None,
                evaluation=evaluation,
            )
            for i in range(len(self.actors))
        ]
        actor_outputs = torch.stack(actor_outputs, dim=1)
        agent_states = TensorDict(
            {
                "rnn_states": actor_outputs["rnn_states"],
                "rnn_states_critic": critic_outputs["rnn_states_critic"],
            },
            batch_size=len(self.envs),
            device=self.device,
        )
        return actor_outputs, critic_outputs, agent_states

    def initialize_agent_state(self, batch_size: int) -> TensorDict:
        agent_states = TensorDict(
            {
                "rnn_states": torch.zeros(
                    batch_size,
                    self.n_agents,
                    self.config.actor.hidden_dim,
                ),
                "rnn_states_critic": torch.zeros(
                    batch_size,
                    self.n_agents,
                    self.config.critic.hidden_dim,
                ),
            },
            batch_size=batch_size,
            device=self.device,
        )
        return agent_states
