import torch 
from fullboard.MinesweeperBoard import MinesweeperDiscrete

TEST_EPISODES = 100
PATH = './trained_model_300000.pt'
model = torch.load(PATH)

env = MinesweeperDiscrete()

games_won = 0
for episode in range(TEST_EPISODES):
    obs, _ = env.reset()
    done = False
    while not done:
        # Convert the state to the appropriate format (e.g., tensor or numpy array)
        #print(env.showed_board)
        
        action = model.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        # if reward == -10:
        #     print("BOOM")
        
        if reward == 10:
            print("WIN")
            games_won += 1
        # Pass the state through the model to get action probabilities
        done = terminated or truncated

print("TEST FINISHED. GAMES WON:", games_won)