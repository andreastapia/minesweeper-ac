import config
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from fullboard.MinesweeperBoard import MinesweeperDiscrete
from fullboard.ActorCriticAgentBoard import ActorCriticAgent

rng = np.random.default_rng()
plt.rcParams["figure.figsize"] = (10, 5)

def running_average(x, N):
    cumsum = np.cumsum(np.insert(np.array(x), 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

def main():

    env = MinesweeperDiscrete()

    total_num_episodes = int(5e5)  # Total number of episodes

    #input_channels, conv_hidden, output_channels, learning_rate, gammaS
    agent = ActorCriticAgent(1,128,81,0.001,0.99)

    rewards = []
    last_1k = []
    steps = []
    games_won = 0
    for episode in range(total_num_episodes):
        #print(episode)
        episode_steps = 0
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            if reward == config.WIN_REWARD:
                games_won += 1
            #print(env.showed_board)
            #print(reward)
            agent.rewards.append(reward)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

        #print("OUTPUT WEIGHTS CRITIC", agent.critic.out_layer.weight.data)
        agent.update()
        rewards.append(episode_reward)
        steps.append(episode_steps)
        last_1k.append(episode_reward)

        if episode % 1000 == 0:
            avg_reward = np.mean(rewards)
            avg_steps = np.mean(steps)
            print("Episode:", episode, "Average Reward:", avg_reward, "Current Average Steps:", avg_steps)
            print("LAST 1K AVERAGE REWARD", np.mean(last_1k))
            print("EPISODE", episode, "REWARD", episode_reward, "STEPS", episode_steps)
            
            last_1k = []

        if episode % 10000 == 0:            
            print("WIGHT SHAPE", agent.critic.out_layer.weight.data.shape)
            print("OUTPUT WEIGHTS CRITIC", agent.critic.out_layer.weight.data)
            print("MAXIMUM WEIGHT", agent.critic.out_layer.weight.data.max())
            print("GAMES WON", games_won)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if episode % 100000 == 0 and episode != 0:
            filename = "trained_model_{episode}.pt"
            torch.save(agent, filename.format(episode=episode))

    # # Guardar
    torch.save(agent, 'trained_model.pt')

    plt.figure(figsize=(15, 6))
    plt.subplot(221)
    plt.plot(rewards)
    plt.plot(running_average(reward, 1000))
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.subplot(222)
    plt.plot(steps)
    plt.plot(running_average(steps, 1000))
    plt.xlabel("Episodes")
    plt.ylabel("steps")
    plt.subplot(223)
    plt.plot(agent.get_critic_training_loss())
    plt.plot(running_average(agent.get_critic_training_loss(), 1000))
    plt.xlabel("Episodes")
    plt.ylabel("Critic MSE Loss")
    plt.subplot(224)
    plt.plot(agent.get_actor_training_loss())
    plt.plot(running_average(agent.get_actor_training_loss(), 1000))
    plt.xlabel("Episodes")
    plt.ylabel("Actor Loss")
    plt.show()
    plt.show()


if __name__ == '__main__':
    main()