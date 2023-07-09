from Minesweeper import MinesweeperEnvironment
from ActorCriticAgent import ActorCriticAgent

def main():
    # Example usage of the Actor-Critic agent for Minesweeper
    env = MinesweeperEnvironment()  # Your Minesweeper environment implementation
    input_channels = 1  # Number of channels in the input grid representation
    hidden_size = 64
    output_size = env.get_action_size()
    learning_rate = 0.001
    gamma = 0.99
    num_episodes = 1000

    agent = ActorCriticAgent(input_channels, hidden_size, output_size, learning_rate, gamma)

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.rewards.append(reward)
            state = next_state

        agent.update()

        # Print episode stats
        total_reward = sum(agent.rewards)
        print(f"Episode {episode + 1} | Total Reward: {total_reward}")

    # Use the trained agent to play Minesweeper
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state

    env.render()

if __name__ == '__main__':
    main()