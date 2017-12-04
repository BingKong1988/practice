import numpy as np
import sys
if '../' not in sys.path:
    sys.path.append('../')
from lib.envs.blackjack import BlackjackEnv

env = BlackjackEnv()

def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print('Player score : {} (Usable ace: {}), Dealer score: {}'.format(score, usable_ace, dealer_score))

def strategy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >=20 else 1

if __name__ == '__main__':
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            print_observation(observation)
            action = strategy(observation)
            print('Taking action: {}'.format(['Stick', 'Hit'][action]))
            observation, reward, done, _ = env.step(action)
            if done:
                print_observation(observation)
                print('Game end. Reward: {}\n'.format(float(reward)))
                break
