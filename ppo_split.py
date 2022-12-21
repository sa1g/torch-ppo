"""
Temporary single file application
"""
# Import modules
from abc import ABC
import datetime
import random

import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def running_average(x, n):
    N = n
    kernel = np.ones(N)
    conv_len = x.shape[0] - N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i + N]  # matrix multiplication operator: np.mul
        y[i] /= N
    return y

class Actor(nn.Module):
    """
    Actor-Critic model, also known as Policy-Value Function network
    """

    def __init__(self, state_dim, action_dim, learning_rate=0.0003):
        super().__init__()
        # Set values
        self.state_dim = torch.tensor(state_dim)
        self.action_dim = torch.tensor(action_dim)
        self.learning_rate = learning_rate

        # Common NN layers
        self.linear1 = nn.Linear(self.state_dim, 64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        # Actor
        self.actor0 = nn.Linear(32, self.action_dim)

        # # Critic
        # self.critic = nn.Linear(32, 1)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

    def forward(self, observation):
        """
        Returns policy_probs, vf_reward
        """
        # Forward OBS into every layer
        x = torch.tensor(observation)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)

        # Get policy_prob
        policy_probs = self.actor0(x)

        # FIXME: this type of softmax is slow and doesnt scale well.
        return torch.nn.functional.softmax(policy_probs, dim=-1)

    def predict(self, observation):
        """
        Returns policy_act, policy_probs, vf_reward
        """
        policy_probs = self.forward(observation=observation)

        policy_act = np.random.choice(np.arange(self.action_dim),
                                      p=policy_probs.detach().numpy())

        return policy_act, policy_probs

class Critic(nn.Module):
    """
    Actor-Critic model, also known as Policy-Value Function network
    """

    def __init__(self, state_dim, action_dim, learning_rate=0.0003):
        super().__init__()
        # Set values
        self.state_dim = torch.tensor(state_dim)
        self.action_dim = torch.tensor(action_dim)
        self.learning_rate = learning_rate

        # Common NN layers
        self.linear1 = nn.Linear(self.state_dim, 64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        # Critic
        self.critic = nn.Linear(32, 1)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

    def forward(self, observation):
        """
        Returns policy_probs, vf_reward
        """
        # Forward OBS into every layer
        x = torch.tensor(observation)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)

        # Get vf_reward
        vf_reward = self.critic(x)

        # FIXME: this type of softmax is slow and doesnt scale well.
        return vf_reward

    def predict(self, observation):
        """
        Returns policy_act, policy_probs, vf_reward
        """
        vf_reward = self.forward(observation=observation)

        return vf_reward

class Model:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim)

    def predict(self, observation):
        policy_act, policy_probs = self.actor.predict(observation=observation)
        vf_reward = self.critic.predict(observation=observation)
        return policy_act, policy_probs, vf_reward

    def safe_ratio(self, num, den):
        """
        Returns 0 if nan, else value

        -G
        """
        return num/(den+1e-10) * (torch.abs(den)>0)

    def learn(self, observation, policy_act, policy_prob, vf_reward, reward, iteration):
        """
        GAE + PPO LOSS w/clip, entropy and vf_loss;
        entropy is with mean, not sum bf of different batch sizes.
        """

        # GAE Constants:
        gae_gamma = 0.99
        gae_lambda = 0.95
        
        # PPO Hyperparamters:
        policy_clip = 0.2
        c1 = 1
        c2 = 0.01

        new_policy_act, new_policy_prob, new_vf_reward = [], [], []

        # 1. Get new policy FORWARD results
        for obs in observation:
            pa, pp, vfr = self.predict(observation=torch.tensor(obs))
            new_policy_act.append(pa)
            new_policy_prob.append(pp)
            new_vf_reward.append(vfr)

        # 2. Calculate GAE
        deltas = [
            r + gae_gamma * nv - v
            for r, nv, v in zip(reward, new_vf_reward, vf_reward )
        ]

        deltas_len = len(deltas)
        deltas = torch.stack(deltas)

        for t in reversed(range(deltas_len -1)):
            deltas[t] = deltas[t] + gae_gamma * gae_lambda * deltas[t+1]

        notnormalized_deltas = torch.squeeze(deltas)

            # 2.1 Normalize GAE, returns advantage
        deltas = (deltas - deltas.mean()) / (deltas.std() + 1e-8)
        advantage = torch.squeeze(deltas)

        # 3. Policy loss

        ## Get policy_prob and new_policy_prob for action taken in policy_act with policy_prob probabilities.
        policy_a = []
        new_policy_a = []
        for policy, new_policy, action in zip(policy_prob, new_policy_prob, policy_act):
            """
            We don't have problems with action mask 'nones' because it's always the same, so it's impossibile
            having a smth/0 or smth/-inf or smth like this.
            """
            policy_a.append(torch.squeeze(policy)[action])
            new_policy_a.append(torch.squeeze(new_policy)[action])
   
        policy_prob = torch.tensor(policy_a)
        new_policy_prob = torch.tensor(new_policy_a)

        ## Calculate probability ratio
        ## PAPER: (6-)
        prob_ratio = self.safe_ratio(new_policy_prob, policy_prob)

        ## Calcualte first argument of L_CLIP ( r_t(\theta)*\hat{A_t})
        ## PAPER: (6) \hat{E_t} excluded
        weighted_probs = advantage * prob_ratio
        ## PAPER: (7) -> only clip part
        weighted_clipped_probs = torch.clamp(prob_ratio, 1-policy_clip, 1+policy_clip)
        ## PAPER: (7) L_CLIP
        policy_loss = torch.min(weighted_probs, weighted_clipped_probs).mean()

        # 4. Value function loss
        ## PAPER: (9), L_t^{VF}
        target = notnormalized_deltas + torch.tensor(vf_reward)
        vf_loss = torch.nn.functional.mse_loss(target, torch.tensor(new_vf_reward))

        # 5. Entropy
        ## PAPER: (9), S_t
        ## Shannon entropy where SUM -> MEAN so that it doesn't depend on "batch size"
        ## FIXME: 1e-10 ?
        entropy = -(new_policy_prob * torch.log(new_policy_prob + 1e-10))
        entropy = torch.mean(entropy)

        # 6. Total Loss
        ## PAPER: (9)
        ## Inverted sign because torch.optimizer.Adam is a SGD and we need a SGA, so if we invert the sign an SGD works correctly
        total_loss = -(policy_loss - c1*vf_loss + c2*entropy)
        
        if iteration % 100 == 0:
            print(total_loss.item())

        # 7. Backpropagation
        ## THIS IS DIFFERENT

        # 7.1 Policy
        # self.actor.optimizer.zero_grad()
        # actor_loss = -(policy_loss + c2*entropy)
        # actor_loss.backward(retain_graph=True)
        # self.actor.optimizer.step()

        # 7.2 Value Function
        self.critic.optimizer.zero_grad()
        vf_loss.backward()
        self.critic.optimizer.step()

        return total_loss, policy_loss, vf_loss, entropy

def batch_and_train(env):
    """
    Performs batching and training for each batch
    """
    # Define model and constants
    model = Model(env.observation_space.shape[0], env.action_space.n)
    EPISODE_LENGTH = 300
    MAX_EPISODES = 1000
    VISUALIZE_STEP = 100
    score = []

    total_loss, policy_loss, vf_loss, entropy = [],[],[],[]

    for episode in range(MAX_EPISODES):
        # As these episodes are defined by each `done` we can actually run multiple episodes.

        obs = env.reset()
        done = False
        
        mem_observation = []
        mem_policy_act = []
        mem_policy_prob = []
        mem_reward = []
        mem_vf_reward = []

        episode_score = 0

        for t in range(EPISODE_LENGTH):
            # BATCHING
            # Run the episode for max `EPISODE_LENGTH` steps and register its data in the `episode_memory`
            # if the episode is `done` the batch is interrupted (and training continues)
            policy_act, policy_prob, vf_reward = model.predict(obs)
            new_obs, rew, done, info = env.step(policy_act)
            episode_score += rew

            mem_observation.append(obs)
            mem_policy_act.append(policy_act)
            mem_policy_prob.append(policy_prob)
            mem_reward.append(rew)
            mem_vf_reward.append(vf_reward)

            obs = new_obs

            if done:
                break

        score.append(episode_score)
        
        # Training
        total_los, policy_los, vf_los, entrop = model.learn(mem_observation, mem_policy_act, mem_policy_prob, mem_vf_reward, mem_reward, episode)
            # inside model.learn do gae and other stuff.

        total_loss.append(total_los.item())
        policy_loss.append(policy_los.item())
        vf_loss.append(vf_los.item())
        entropy.append(entrop.item())
        
         # print the status after every VISUALIZE_STEP episodes
        if episode % VISUALIZE_STEP == 0 and episode > 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                episode, np.mean(score[-VISUALIZE_STEP:-1])))
            # domain knowledge applied to stop training: if the average score across last 100 episodes is greater than 195, game is solved
            # if np.mean(score[-100:-1]) > 195:
            #     break

    # Create graphs
    EXPERIMENT_NAME = datetime.datetime.now()

    score = np.array(score)
    avg_score = running_average(score, 100)
    plt.figure(figsize=(15, 7))
    plt.ylabel("Episodic Reward", fontsize=12)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.plot(score, color='gray', linewidth=1, label='score')
    plt.plot(avg_score, color='blue', linewidth=3, label='average score')
    plt.scatter(np.arange(score.shape[0]), score, color='green', linewidth=0.3)
    plt.title('Reward', fontdict=None, loc='center', pad=None)

    plt.legend()

    plt.savefig(f"img/{EXPERIMENT_NAME}_reward.png")



    plt.figure(figsize=(15, 7))
    plt.ylabel("Episodic loss", fontsize=12)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.plot(total_loss, color='green', label='total loss', linewidth=1)
    plt.plot(policy_loss, color='red', label='policy loss', linewidth=1)
    plt.plot(vf_loss, color='blue', label='vf loss', linewidth=1)
    plt.plot(entropy, color='magenta', label='entropy', linewidth=1)
    plt.title('Total Losses', fontdict=None, loc='center', pad=None)
    plt.legend()

    plt.savefig(f"img/{EXPERIMENT_NAME}_loss.png")
    
    plt.figure(figsize=(15, 7))
    plt.ylabel("Policy loss", fontsize=12)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.plot(policy_loss, color='red', label='policy loss', linewidth=1)
    plt.title('Policy loss', fontdict=None, loc='center', pad=None)
    plt.legend()

    plt.savefig(f"img/{EXPERIMENT_NAME}_policy_loss.png")

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    # Seeding for reproducibility
    # torch.manual_seed(0)
    # random.seed(0)
    # np.random.seed(0)
    # env.seed(0)

    batch_and_train(env=env)
