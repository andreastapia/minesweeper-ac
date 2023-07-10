import torch 

PATH = './trained_model.pt'
model = torch.load(PATH)

TEST_GAMES = 100

for i in range(TEST_GAMES):
    continue