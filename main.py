import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
from Minesweeper import MinesweeperDiscrete

import gymnasium as gym

import torch
import pandas as pd
import seaborn as sns
from reinforce import REINFORCE


plt.rcParams["figure.figsize"] = (10, 5)

def main():
    print("STARTING")
    env = MinesweeperDiscrete()
    # Create and wrap the environment
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    total_num_episodes = int(100000)  # Total number of episodes
    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.observation_space.shape[0] * env.observation_space.shape[1]
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space.n
    rewards_over_seeds = []

    print("OBSERVATION SHAPE", env.observation_space.shape, "ACTION SHAPE", env.action_space.shape)
    print("OBSERVATION", obs_space_dims, "ACTION", action_space_dims)

    for seed in [1]:  # Fibonacci seeds
        # set seed
        #torch.manual_seed(seed)
        #random.seed(seed)
        #np.random.seed(seed)

        # Reinitialize agent every seed
        
        agent = REINFORCE(obs_space_dims, action_space_dims)
        reward_over_episodes = []

        for episode in range(total_num_episodes):
            
            # gymnasium v26 requires users to set seed while resetting the environment
            obs, info = wrapped_env.reset(seed=seed)
            done = False
            ep_reward = 0
            while not done:
                action = agent.act(obs)
                #print("EPISODE", episode, "ACTION", action)
                # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
                # These represent the next observation, the reward from the step,
                # if the episode is terminated, if the episode is truncated and
                # additional info from the step
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards.append(reward)

                # End the episode when either truncated or terminated is true
                #  - truncated: The episode duration reaches max number of timesteps
                #  - terminated: Any of the state space values is no longer finite.
                ep_reward += reward
                done = terminated or truncated
            reward_over_episodes.append(wrapped_env.return_queue[-1])
            agent.update()
            if episode % 100 == 0:
                print("Episode", episode, "Reward", ep_reward)
            if episode % 1000 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print("Episode:", episode, "Average Reward:", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)

    # Guardar
    torch.save(agent, 'trained_model.pt')

    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for Minesweeper"
    )
    plt.show()


if __name__ == '__main__':
    main()