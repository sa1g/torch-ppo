# torch-ppo
Multiple implementations of PPO using `CartPole-v1` and a simple actor-critic model
Instead of GAE, this implementation works with Monte-Carlo estimation.

Inspired heavily by [PPO-Pytorch], (https://github.com/nikhilbarhate99/PPO-PyTorch)

The simplest implementation is `simple_ppo/ppo.py,` and the code is well documented over there.

The project also manages multi-agent environments. The chosen environment is a custom "CartPole-v1" wrapper that contains multiple instances of the environment. This solution is extremely simple; a more complex one can be found at: To Do: Add Another Project of Mine
