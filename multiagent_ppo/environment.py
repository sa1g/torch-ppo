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

    def __init__(self, n_envs: int = 2):
        if n_envs < 1:
            ValueError(f"n_envs is {n_envs}, it must be >= 1")
            sys.exit(f"n_envs is {n_envs}, it must be >= 1")

        self.n_envs = n_envs

        self.envs = {}
        for e in range(n_envs):
            self.envs[str(e)] = gym.make("CartPole-v1")

        self.observation_space = self.envs["0"].observation_space
        self.action_space = self.envs["0"].action_space
        

    """
    Methods :
        close

    Pretty straightforward.
    """

    def get_keys(self)->list:
        return self.envs.keys()

    def seed(self, seed: int):
        """
        Sets the seed of each environment. Starts with `seed`, ends with `seed` + `n_envs`

        Args:
            seed: seed used for seeding.
        """
        for idx, env in self.envs.items():
            env.seed(seed + int(idx))

    def reset(self) -> dict:
        """
        Resets each environment, returns observations for eachone.

        Returns:
            obs: obs dictionary of each environment
        """
        obs = {}
        for idx, env in self.envs.items():
            obs[idx] = env.reset()

        return obs

    def reset_single(self, key:str) -> dict:
        return self.envs[key].reset()
    
    def step(self, actions: dict):
        """
        Returns:
            obs
            rew
            done
        """
        obs, rew, done = {}, {}, {}
        for (idx, env), action in zip(self.envs.items(), actions.values()):
            obs[idx], rew[idx], done[idx], _ = env.step(action)
            # if idx == '0':
            #     env.render()

        return obs, rew, done

    def close(self):
        for env in self.envs.values():
            env.close()

        


if __name__ == "__main__":
    abla = MultiAgentCartPole(n_envs=4)
    abla.seed(1)
    print(abla.reset())

    import torch

    action = torch.tensor([0])
    actions = {
        "0": action,
        "1": action,
        "2": action,
        "3": action,
    }

    print(abla.step(actions))

    abla.close()
