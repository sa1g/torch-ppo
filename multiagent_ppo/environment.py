"""
MultiAgent CartPole-v1 environment wrapper.
"""
# Modules and libraries
import gym


class MultiAgentCartPole:
    """
    Simple environment wrapper for n `CartPole-v1` envs.

    This multi-agent wrapper is inspired from 
    [AI-Economist](https://github.com/salesforce/ai-economist)'s interface: dictionaries.
    It has been developed to understand how to setup PPO for a complex environment with
    multiple agents and, in case, for multiple policies.
    Here it's possible having N>0 agents and 0<M<=N polcies. At max 1 policy <=> 1 agent.
    """

    def __init__(self, n_envs = 2):
        print("here")
        if n_envs < 0:
            AssertionError(f"n_envs is {n_envs}, it must be >= 1")
        print(n_envs)
        self.n_envs = n_envs
        
        envs = [gym.make("CartPole-v1") for e in range(n_envs)]
        print(envs[0].reset())
        # env = gym.make("CartPole-v1")

    """
    Variables:
        observation_space
        action_space

    Methods :
        seed
        reset
        step
        close

    Pretty straightforward.
    """
    
if __name__ == '__main__':
    MultiAgentCartPole(2)
