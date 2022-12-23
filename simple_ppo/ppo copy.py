# Libraries and modules
# pylint: skip-file
import sys
import torch
import gym
import numpy as np

# Define the network
class ActorCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = torch.nn.Sequential(
                            torch.nn.Linear(obs_dim , 64),
                            torch.nn.Tanh(),
                            torch.nn.Linear(64, 64),
                            torch.nn.Tanh(),
                            torch.nn.Linear(64, action_dim),
                            torch.nn.Softmax(dim=-1)
                        )
        
        # critic
        self.critic = torch.nn.Sequential(
                        torch.nn.Linear(obs_dim, 64),
                        torch.nn.Tanh(),
                        torch.nn.Linear(64, 64),
                        torch.nn.Tanh(),
                        torch.nn.Linear(64, 1)
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
    def __init__(self, obs_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
            ])

        self.MseLoss = torch.nn.MSELoss()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs) # to device
            action, action_logprob = self.policy.act(obs=obs)
        
        self.buffer.states.append(obs)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action # .item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio pi_theta / pi_theta_old
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Clear buffer
        self.buffer.clear()

def train():
    '''
             e@@@@@@@@@@@@@@@
        @@@""""""""""
       @" ___ ___________
      II__[w] | [i] [z] |
     {======|_|~~~~~~~~~|
    /oO--000'"`-OO---OO-'
    '''
    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)

    ## HYPERPARAMs
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 2         # set random seed if required (0 = no random seed)
    # FIXME: missing c1, c2
    ## HYPERPARAMs

    env = gym.make('CartPole-v1')
    
    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.n

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # logging bad way
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                ppo_agent.update()

            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward,2)

                print(f"Episode : {i_episode} \t\t Timestep : {time_step} \t\t Average Reward : {print_avg_reward}")

                print_running_reward = 0
                print_running_episodes = 0

            if done: 
                break
        
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
    
    env.close()

if __name__ == '__main__':
    train()