import gym

from policy import ActorCriticPolicy
from ppo import PPO

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    policy = ActorCriticPolicy(env.observation_space.shape[0], env.action_space.n)
    model = PPO(env=env, policy=policy)
    model.learn(1000)