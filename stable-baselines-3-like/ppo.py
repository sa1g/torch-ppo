"""
Temporary single file application
"""
import datetime
import random
import warnings
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=unused-import
# Import modules
from abc import ABC
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar,
                    Union)

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from buffers import RolloutBuffer
from utils import get_device, set_random_seed, obs_as_tensor, update_learning_rate, explained_variance

Schedule = Callable[[float], float]

class PPO():
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    Args:
        policy: The policy model to use
        env: The environment to learn from
        learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        n_steps: The number of steps to run for each environment per update
            (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
            NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
            See https://github.com/pytorch/pytorch/issues/29372
        batch_size: Minibatch size
        n_epochs: Number of epoch when optimizing the surrogate loss
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range: Clipping parameter, it can be a function of the current progress
            remaining (from 1 to 0).
        clip_range_vf: Clipping parameter for the value function,
            it can be a function of the current progress remaining (from 1 to 0).
            This is a parameter specific to the OpenAI implementation. If None is passed (default),
            no clipping will be done on the value function.
            IMPORTANT: this clipping depends on the reward scaling.
        normalize_advantage: Whether to normalize or not the advantage
        ent_coef: Entropy coefficient for the loss calculation
        vf_coef: Value function coefficient for the loss calculation
        max_grad_norm: The maximum value for the gradient clipping
        use_sde: Whether to use generalized State Dependent Exploration (gSDE)
            instead of action noise exploration (default: False)
        sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
            Default: -1 (only sample at the beginning of the rollout)
        target_kl: Limit the KL divergence between updates,
            because the clipping is not enough to prevent large update
            see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
            By default, there is no limit on the kl div.
        tensorboard_log: the log location for tensorboard (if None, no logging)
        policy_kwargs: additional arguments to be passed to the policy on creation
        verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
            debug messages
        seed: Seed for the pseudo random generators
        device: Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible.
        _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy,
        env,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_Setup_model: bool = True,
    ):
        ## TODO: ho disattivato tutto quello che e' vectorized_environment, quindi 
        ## non funziona. Non sarebbe male studiare al volo i vantaggi (immagino batching) e 
        ## implementarlo quando le cose funzionano. Vedremo.
        
        ## FIXME in sb3 eredita da BaseAlgorithm, quindi mancano alcuni metodi.
        self.device = get_device(device)
        self.env = None
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None  # type: Optional[gym.spaces.Space]
        self.action_space = None # type: Optional[gym.spaces.Space]
        self.num_timesteps = 0
        self._total_timesteps = 0
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.action_noise = None
        self.start_time = None
        self.policy = policy # Modified by -E
        self.learning_rage = learning_rate
        self.tensorboard_log = tensorboard_log
        self.lr_schedule = None # type: Optional[Schedule]
        self._last_obs = None # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        self._last_original_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._episode_num = 0
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self._current_progress_remaining = 1
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        self._n_updates = 0  # type: int
        self._logger = None  # type: Logger
        self._custom_logger = False

        if env is not None:
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space

            if self.use_sde and not isinstance(self.action_space, gym.spaces.Box):
                raise ValueError("generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

            if isinstance(self.action_space, gym.spaces.Box):
                assert np.all(
                    np.isfinite(np.array([self.action_space.low, self.action_space.high]))
                ), "Continuous action space must have a finite lower and upper bound"

        ## FIXME in sb3 eredita da onpolicyalgorithms, quindi mancano alcuni metodi.
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda = self.gae_lambda,
        )

        self.set_random_seed(seed=self.seed)
        # FIXME: buffer_class

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"
        
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.n_steps # TODO: * self.env.num_envs
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps})" # and n_envs={self.env.num_envs})"
                )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

    # funzioni che potrebbero tornare utili da base_class
    # _setup_lr_schedule
    # _update_learning_rate
    # 
    def _update_learning_rate(self, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def set_random_seed(self, seed: Optional[int]=None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == torch.device("cuda").type)
        self.action_space.seed(seed)
        # self.env is always a VecEnv
        if self.env is not None:
            self.env.seed(seed)

    def collect_rollouts(
        self,
        env,
        rollout_buffer,
        n_rollout_steps:int
        ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        Args:
            env: The training environment
            rollout_buffer: Buffer to fill with rollouts
            n_rollout_steps: Number of experiences to collect per environment
        Return:
            True if function returned with at least `n_rollout_steps`
                collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # print("HERE")
        cumulative_rewards = 0
        print(n_rollout_steps)
        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            cumulative_rewards += rewards
            n_steps +=1
            self.num_timesteps += 1
            
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, 
                values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones
            # print("b")
            if dones:
                print(dones)
                self._last_obs = env.reset()
        
        print(f"REW: {cumulative_rewards}, steps: {n_steps}")

        with torch.no_grad():
            # Computevalue for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        # print("pre adv")
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        # print("aft adv")
        return True

    def learn(
        self, 
        total_timesteps: int,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        ):
        self._last_obs = self.env.reset()

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(env = self.env, rollout_buffer= self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            self.train()

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        # clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        # if self.clip_range_vf is not None:
            # clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        clip_range = 0.2
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        
        # explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        print(f"")

        # Logs
        # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        # self.logger.record("train/value_loss", np.mean(value_losses))
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        # self.logger.record("train/loss", loss.item())
        # self.logger.record("train/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.record("train/clip_range_vf", clip_range_vf)


