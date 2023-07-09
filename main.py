import torch
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from Minesweeper import MinesweeperDiscrete
from ActorCriticAgent import ActorCriticAgent

rng = np.random.default_rng()
plt.rcParams["figure.figsize"] = (10, 5)

def main():
    print("STARTING")
    env = MinesweeperDiscrete()
    # Create and wrap the environment
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    total_num_episodes = int(50000)  # Total number of episodes
    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.observation_space.shape[0] * env.observation_space.shape[1]
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space.n
    rewards_over_seeds = []

    obs, _ = wrapped_env.reset(seed=1)

    agent = ActorCriticAgent(1,32,81,0.005,0.99)


    print("OBSERVATION SHAPE", env.observation_space.shape, "ACTION SHAPE", env.action_space.shape)
    print("OBSERVATION", obs_space_dims, "ACTION", action_space_dims)

    reward_over_episodes = []
    for episode in range(total_num_episodes):
        episode_steps = 0
        obs, info = wrapped_env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = wrapped_env.step(action)
            agent.rewards.append(reward)
            ep_reward += reward
            episode_steps += 1
            done = terminated or truncated
        #print("EPISODE", episode, "STEPS", episode_steps, "REWARD", wrapped_env.return_queue[-1])

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

        agent.update()
        reward_over_episodes.append(wrapped_env.return_queue[-1])

    rewards_over_seeds.append(reward_over_episodes)

    # # Guardar
    #torch.save(agent, 'trained_model.pt')
    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="Actor-Critic for Minesweeper"
    )
    plt.show()


if __name__ == '__main__':
    main()