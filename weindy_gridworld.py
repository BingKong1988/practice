import gym
import numpy as np
import sys
if '../' not in sys.path:
	sys.path.append('../')
from lib.envs.windy_gridworld import WindyGridworldEnv

env = WindyGridworldEnv()
print(env.reset())
env.render()
print(env.step(1))
env.render()
