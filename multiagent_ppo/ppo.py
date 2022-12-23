# Libraries and modules
# pylint: skip-file
import sys
import torch
import numpy as np
from environment import MultiAgentCartPole

# Define the network
class ActorCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, action_dim),
            torch.nn.Softmax(dim=-1),
        )

        # critic
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, obs):
        """
        Args:
            obs: agent environment observation

        Returns:
            action: taken action
            action_logprob: log probability of that action
        """
        action_probs = self.actor(obs)
        dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, obs, act):
        """
        Args:
            obs: agent environment observation
            act: action that is mapped with

        Returns:
            action_logprobs: log probability that `act` is taken with this model
            state_values: value function reward prediction
            dist_entropy: entropy of actions distribution
        """
        action_probs = self.actor(obs)
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(act)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs)

        return action_logprobs, state_values, dist_entropy

# FIXME
class RolloutBuffer:
    """
    Buffer used to store batched data
    """

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO:
    # ADD n_agents
    def __init__(
        self, obs_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, keys
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.keys : list = keys
        self.policy = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.MseLoss = torch.nn.MSELoss()

    # ADD multi-agent management
    def select_action(self, obs):
        actions, actions_logprob = {}, {}

        for key in self.keys:
            

            with torch.no_grad():
                actions[key], actions_logprob[key] = self.policy.act(obs=obs[key])

        return actions, actions_logprob

    # OKAY
    def update(self, buffer: RolloutBuffer):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(
            reversed(buffer.rewards), reversed(buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio pi_theta / pi_theta_old
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def dict_to_tensor_dict(obs:dict)-> dict:
    out_obs = {}
    for key, value in obs.items():
        out_obs[key]=torch.FloatTensor(value)
    return out_obs

def train():
    '''
             e@@@@@@@@@@@@@@@
        @@@""""""""""
       @" ___ ___________
      II__[w] | [i] [z] |
     {======|_|~~~~~~~~~|
    /oO--000'"`-OO---OO-'
    '''
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 2  # set random seed if required (0 = no random seed)
    # FIXME: missing c1, c2
    
    n_envs = 1
    env : MultiAgentCartPole = MultiAgentCartPole(n_envs=n_envs)
    keys = env.get_keys()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    ppo_agent = PPO(
        state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, keys
    )
    # As this environment doesn't need or care about lstm or stuff like that
    # I can (?) append all data direcly, without any order.
    buffer = RolloutBuffer()

    i_episode = 0
    epochs = 1000
    batch_size = 1000*n_envs

    epoch = 0
    while epoch < epochs:
        state = (dict_to_tensor_dict(env.reset()))
        total_rew = 0
        i_episode = 0
        batch = 0
        # batching

        while batch < batch_size:

            # select action with policy
            action, action_logprob = ppo_agent.select_action(state)

            for key in keys:
                buffer.states.append(state[key])
                buffer.actions.append(action[key])
                buffer.logprobs.append(action_logprob[key])

            state, reward, done = env.step(action)
            state =(dict_to_tensor_dict(state))
            # Saving reward and is_terminals
            
            for key in keys:        
                buffer.rewards.append(reward[key])
                buffer.is_terminals.append(done[key])

            batch += n_envs
            
            for value in reward.values():
                total_rew+=value

            if done:
                for key, value in done.items():
                    if value == True:
                        state[key] = torch.FloatTensor(env.reset_single(key))
                        i_episode += 1

        epoch += 1

        print(f"EPOCH {epoch}, AVG: {total_rew/i_episode}")
        ppo_agent.update(buffer)
        buffer.clear()

    env.close()


if __name__ == "__main__":
    train()
