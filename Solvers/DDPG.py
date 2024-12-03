# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from torch.distributions.normal import Normal

from Solvers.Abstract_Solver import MonitorAbstractSolver
from lib import plotting


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim + act_dim] + hidden_sizes + [1]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.act_lim = act_lim
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.act_lim * F.tanh(self.layers[-1](x))
    
class RewardNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [1]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = obs
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
        super().__init__()
        self.q = QNetwork(obs_dim, act_dim, hidden_sizes)
        self.pi = PolicyNetwork(obs_dim, act_dim, act_lim, hidden_sizes)


class DDPG(MonitorAbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # Create actor-critic network
        self.obs_dim = env.observation_space["env"].shape[0]
        self.act_dim = env.action_space["env"].shape[0]
        self.lim = env.action_space["env"].high[0]
        self.mon_obs_dim = env.observation_space["mon"].n    # Discrete
        self.mon_act_dim = env.action_space["mon"].n         # Discrete
        self.mon_high = np.array([1.0] * self.mon_act_dim)
        # print(self.obs_dim, self.act_dim, self.high, self.mon_obs_dim, self.mon_act_dim)

        self.actor_critic = ActorCriticNetwork(
            self.obs_dim,
            self.act_dim,
            self.lim,
            self.options.layers,
        )
        self.mon_actor_critic = ActorCriticNetwork(
            self.obs_dim + self.mon_obs_dim,
            self.mon_act_dim,
            self.lim,
            self.options.layers,
        )
        # Create target actor-critic network
        self.target_actor_critic = deepcopy(self.actor_critic)
        self.target_mon_actor_critic = deepcopy(self.mon_actor_critic)

        self.reward_model = RewardNetwork(
            self.obs_dim,
            self.options.layers
        )

        self.optimizer_q = Adam(self.actor_critic.q.parameters(), lr=self.options.alpha)
        self.optimizer_pi = Adam(
            self.actor_critic.pi.parameters(), lr=self.options.alpha
        )
        self.optimizer_mq = Adam(self.mon_actor_critic.q.parameters(), lr=self.options.alpha)
        self.optimizer_mpi = Adam(
            self.mon_actor_critic.pi.parameters(), lr=self.options.alpha
        )
        self.optimizer_rwd = Adam(self.reward_model.parameters(), lr=self.options.alpha)

        # Freeze target actor critic network parameters
        for param in self.target_actor_critic.parameters():
            param.requires_grad = False
        for param in self.target_mon_actor_critic.parameters():
            param.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=options.replay_memory_size)

    def convert_from_dict_state(self, state):
        mon_state = np.zeros(self.mon_obs_dim, dtype=np.float32)
        mon_state[state['mon']] = 1
        state = np.concatenate((state['env'], mon_state), axis=None)
        return state
    
    def convert_to_dict_action(self, action):
        mon_action = np.argmax(action[-self.mon_act_dim : ])
        env_action = action[:self.act_dim]
        return {
            "mon": mon_action,
            "env": env_action
        }

    @torch.no_grad()
    def update_target_networks(self, tau=0.995):
        """
        Copy params from actor_critic to target_actor_critic using Polyak averaging.
        """
        for param, param_targ in zip(
            self.actor_critic.parameters(), self.target_actor_critic.parameters()
        ):
            param_targ.data.mul_(tau)
            param_targ.data.add_((1 - tau) * param.data)
        
        for param, param_targ in zip(
            self.mon_actor_critic.parameters(), self.target_mon_actor_critic.parameters()
        ):
            param_targ.data.mul_(tau)
            param_targ.data.add_((1 - tau) * param.data)

    def create_greedy_policy(self):
        """
        Creates a greedy policy.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        @torch.no_grad()
        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            env_state = state[:self.obs_dim]
            return self.actor_critic.pi(env_state).numpy()

        return policy_fn

    @torch.no_grad()
    def select_action(self, state):
        """
        Selects an action given state.

         Returns:
            The selected action (as an int)
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        mu1 = self.actor_critic.pi(state[:self.obs_dim])
        mu2 = self.mon_actor_critic.pi(state)
        mu = torch.cat([mu1, mu2], dim=-1)
        m = Normal(
            torch.zeros(self.act_dim + self.mon_act_dim),
            torch.ones(self.act_dim + self.mon_act_dim),
        )
        noise_scale = 0.1
        action_limit = self.lim
        action = mu + noise_scale * m.sample()
        return torch.clip(
            action,
            -action_limit,
            action_limit,
        ).numpy()

    @torch.no_grad()
    def compute_target_values(self, next_states, rewards, dones, critic):
        actions = critic.pi(next_states)
        return rewards + self.options.gamma * (1 - dones) * critic.q(next_states, actions)

    def update_reward_network(self, next_state, reward):
        reward = torch.tensor(reward)
        loss = (self.reward_model(next_state) - reward) ** 2
        self.optimizer_rwd.zero_grad()
        loss.backward()
        self.optimizer_rwd.step()


    def replay(self):
        """
        Samples transitions from the replay memory and updates actor_critic network.
        """
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            minibatch = [
                np.array(
                    [
                        transition[idx]
                        for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(6)
            ]
            states, actions, rewards, env_rewards, next_states, dones = minibatch
            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            env_rewards = torch.as_tensor(env_rewards, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            env_states = states[:, :self.obs_dim]
            # Current Q-values
            current_q = self.actor_critic.q(env_states, actions[:, :self.act_dim])
            current_mq = self.mon_actor_critic.q(states, actions[:, self.act_dim:])
            # Target Q-values
            target_q = self.compute_target_values(env_states, env_rewards, dones, self.actor_critic)
            target_mq = self.compute_target_values(next_states, rewards, dones, self.mon_actor_critic)

            # Optimize critic network
            loss_q = self.q_loss(current_q, target_q).mean()
            self.optimizer_q.zero_grad()
            loss_q.backward()
            self.optimizer_q.step()

            loss_mq = self.q_loss(current_mq, target_mq).mean()
            self.optimizer_mq.zero_grad()
            loss_mq.backward()
            self.optimizer_mq.step()

            # Optimize actor network
            loss_pi = self.pi_loss(env_states, self.actor_critic).mean()
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()

            loss_mpi = self.pi_loss(states, self.mon_actor_critic).mean()
            self.optimizer_mpi.zero_grad()
            loss_mpi.backward()
            self.optimizer_mpi.step()

            # # Optimize reward network
            # current_rewards = self.reward_model(states[:, :self.obs_dim])
            # loss_r = ((current_rewards - env_rewards) ** 2).mean()
            # self.optimizer_rwd.zero_grad()
            # loss_r.backward()
            # self.optimizer_rwd.step()


    def memorize(self, state, action, reward, env_reward, next_state, done):
        """
        Adds transitions to the replay buffer.
        """
        self.replay_memory.append((state, action, reward, env_reward, next_state, done))

    def train_episode(self):
        """
        Runs a single episode of the DDPG algorithm.

        Use:
            self.select_action(state): Sample an action from the policy.
            self.step(action): Performs an action in the env.
            self.memorize(state, action, reward, next_state, done): store the transition in
                the replay buffer.
            self.replay(): Sample transitions and update actor_critic.
            self.update_target_networks(): Update target_actor_critic using Polyak averaging.
        """

        state, _ = self.env.reset()
        state = self.convert_from_dict_state(state)
        for _ in range(self.options.steps):
            action = self.select_action(state)
            next_state, _reward, done, _ = self.step(state, action)
            if np.isnan(_reward["proxy"]):
                _reward["proxy"] = self.reward_model(torch.as_tensor(next_state['env'])).detach().numpy()
                # print("approx: ", _reward["proxy"])
            else:
                self.update_reward_network(torch.as_tensor(next_state['env'], dtype=torch.float32), 
                                           torch.as_tensor(_reward["proxy"], dtype=torch.float32))
            reward = _reward["proxy"] + _reward["mon"]
            next_state = self.convert_from_dict_state(next_state)
            self.memorize(state, action, reward, _reward["proxy"], next_state, done)
            self.replay()
            self.update_target_networks()
            state = next_state
            if done:
                break
            

    def q_loss(self, current_q, target_q):
        """
        The q loss function.

        args:
            current_q: Current Q-values.
            target_q: Target Q-values.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        return  (current_q - target_q) ** 2

    def pi_loss(self, states, critic):
        actions = critic.pi(states)
        return - critic.q(states, actions)

    def __str__(self):
        return "DDPG"

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)
