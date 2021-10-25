from gym_lattice.envs import Lattice2DEnv
from gym import spaces
import numpy as np

np.random.seed(42)

seq = 'HHPHH' # Our input sequence
action_space = spaces.Discrete(4) # Choose among [0, 1, 2 ,3]
env = Lattice2DEnv(seq)

for i_episodes in range(5):
    env.reset()
    while not env.done:
        # Random agent samples from action space
        action = action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Episode finished! Reward: {} | Collisions: {} | Actions: {}".format(reward, info['collisions'], info['actions']))
            break
