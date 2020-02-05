import gym
import random
import torch
import numpy as np
from ddpg import ddpg
    
    
if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    #env.seed(0)
    
    agent = ddpg()
    
    #agent.actor.load_state_dict(torch.load('actor.pkl'))
    agent.actor.load_state_dict(torch.load('smart_actor.pkl'))

    for i in range(10):
        total = 0
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total += reward
                
        print(total)
    env.close()