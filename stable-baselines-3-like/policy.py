from typing import Any, Dict, Optional, Tuple
import torch
import gym.spaces as spaces

class ActorCriticPolicy(torch.nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate=0.0003):
        super().__init__()
        # Set values
        self.state_dim = torch.tensor(state_dim)
        self.action_dim = torch.tensor(action_dim)
        self.learning_rate = learning_rate

        self.observation_space = self.state_dim
        self.action_space = self.action_dim

        # Common NN layers
        self.linear1 = torch.nn.Linear(self.state_dim, 64)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, 32)
        self.relu2 = torch.nn.ReLU()

        # Actor
        self.actor0 = torch.nn.Linear(32, self.action_dim)

        # Critic
        self.critic = torch.nn.Linear(32, 1)

        self.optimizer = torch.optim.Adam(self.parameters(),
                      lr=self.learning_rate)

        self.action_dist = torch.distributions.Categorical

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        x = self.linear1(obs)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)

        latent_pi = self.actor0(x)
        values = self.critic(x)
        
        # Evaluate the values for the given observations
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.sample() # this is deterministic, otherwise self.sample()
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> torch.distributions.Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        # mean_actions = self.action_net(latent_pi)
        return self.action_dist(logits=latent_pi)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        x = self.linear1(obs)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)

        return self.critic(x)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        x = self.linear1(obs)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        
        latent_pi = self.actor0(x)
        values = self.critic(x)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    # def make_proba_distribution(
    #     action_space: spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
    # ) -> Distribution:
    #     """
    #     Return an instance of Distribution for the correct type of action space

    #     :param action_space: the input action space
    #     :param use_sde: Force the use of StateDependentNoiseDistribution
    #         instead of DiagGaussianDistribution
    #     :param dist_kwargs: Keyword arguments to pass to the probability distribution
    #     :return: the appropriate Distribution object
    #     """
    #     if dist_kwargs is None:
    #         dist_kwargs = {}

        
    #     return CategoricalDistribution(action_space.n, **dist_kwargs)
        