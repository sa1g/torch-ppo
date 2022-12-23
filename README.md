# torch-ppo
Multiple implementations of PPO using `CartPole-v1` and a simple Actor-Critic model.
Instead of GAE this implementation works with Monte-Carlo Estimation.

Strongly inspired by [PPO-Pytorch](https://github.com/nikhilbarhate99/PPO-PyTorch)

The simplet impementation is `simple_ppo/ppo.py`, code is well documented over there.

The project also manages Multi-Agent environments. The selected env is a custom made `CartPole-v1` wrapper with multiple instances of the env. This solutin is extemely simple, a more complex one can be found at: TODO ADD OTHER PROJECT OF MINE

