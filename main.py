import numpy as np
import gym
import torch
import random
from ddpg import ddpg
from noise import OUNoise


if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")

    state_dim = 2
    action_dim = 1

    seed = 228
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gamma = 0.99
    max_episodes = 50
    buffer_maxlen = 5000
    batch_size = 640
    critic_lr = 1e-3
    actor_lr = 1e-4
    tau = 1e-3

    agent = ddpg(state_dim, action_dim, gamma, tau, buffer_maxlen, batch_size, critic_lr, actor_lr)
    noise = OUNoise(env.action_space)

    step = 0

    for episode in range(max_episodes):
        state = env.reset()
        total = 0
        done = False
        while True:
            action = agent.act(state)
            action = noise.get_action(action, step)
            
            next_state, reward, done, _ = env.step(action)
            total += reward
            
            if next_state[0] > 0.0:
                reward += 10
            reward += 50 * abs(next_state[1])
            
            transition = state, action, reward, next_state, done

            agent.update(transition)   

            if done:
                print(f'episode {episode} done. reward: {total}')
                agent.save()
                break

            state = next_state
            step += 1

    print('Training is over!')
    print("Let's evaluate our model!")
    
    for _ in range(10):
            state = env.reset()
            total = 0
            done = False
            while True:
                env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                total += reward  

                if done:
                    print(f'{total}')
                    break

                state = next_state
