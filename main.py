import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from Minesweeper import MinesweeperDiscrete
from ActorCriticAgent import ActorCriticAgent

rng = np.random.default_rng()
plt.rcParams["figure.figsize"] = (10, 5)

def running_average(x, N):
    cumsum = np.cumsum(np.insert(np.array(x), 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

def main():

    env = MinesweeperDiscrete()

    total_num_episodes = int(20000)  # Total number of episodes

    #input_channels, conv_hidden, output_channels, learning_rate, gammaS
    agent = ActorCriticAgent(11,64,81,0.001,0.99)

    rewards = []
    steps = []
    for episode in range(total_num_episodes):
        #print(episode)
        episode_steps = 0
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.act(obs)
            #print("EPISODE", episode, "ACTION", action)
            obs, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated
       
        agent.update()
        rewards.append(episode_reward)
        steps.append(episode_steps)

        if episode % 1000 == 0:
            avg_reward = int(np.mean(rewards))
            avg_steps = int(np.mean(steps))
            print("Episode:", episode, "Average Reward:", avg_reward, "Current Average Steps:", avg_steps)

        if episode % 10000 == 0 and episode != 0:
            filename = "trained_model_{episode}.pt"
            torch.save(agent, filename.format(episode=episode))

    # # Guardar
    torch.save(agent, 'trained_model.pt')

    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.plot(rewards)
    plt.plot(running_average(reward, 1000))
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.subplot(122)
    plt.plot(steps)
    plt.plot(running_average(steps, 1000))
    plt.xlabel("Episodes")
    plt.ylabel("steps")
    plt.show()
    plt.show()


if __name__ == '__main__':
    main()