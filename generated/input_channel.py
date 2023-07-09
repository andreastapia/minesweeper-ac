import torch

matrix = torch.tensor([[1, 4, 7, 2, -2, 5, 8, 0, 3],
                       [2, 6, 0, 8, 1, 3, 5, 4, 7],
                       [3, 1, 7, 2, 4, 6, 0, 8, 5],
                       [2, 5, 1, 7, 4, 3, 6, 8, 0],
                       [0, 3, 6, 2, 5, 1, 8, 7, 4],
                       [8, 2, 0, 4, 6, 7, 1, 3, 5],
                       [4, 7, 3, 5, 0, 1, 8, 2, 6],
                       [6, 8, 5, 3, 7, 0, 2, 1, 4],
                       [1, 0, 4, 6, 2, 5, 7, 3, 8]])

min_value = -2
max_value = 8
normalized = (matrix - min_value) / (max_value - min_value)
normalized = normalized.unsqueeze(0).unsqueeze(0)

print(normalized)
