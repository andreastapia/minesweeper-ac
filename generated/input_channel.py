# import torch

# matrix = torch.tensor([[1, 4, 7, 2, -2, 5, 8, 0, 3],
#                        [2, 6, 0, 8, 1, 3, 5, 4, 7],
#                        [3, 1, 7, 2, 4, 6, 0, 8, 5],
#                        [2, 5, 1, 7, 4, 3, 6, 8, 0],
#                        [0, 3, 6, 2, 5, 1, 8, 7, 4],
#                        [8, 2, 0, 4, 6, 7, 1, 3, 5],
#                        [4, 7, 3, 5, 0, 1, 8, 2, 6],
#                        [6, 8, 5, 3, 7, 0, 2, 1, 4],
#                        [1, 0, 4, 6, 2, 5, 7, 3, 8]])

# min_value = -2
# max_value = 8
# normalized = (matrix - min_value) / (max_value - min_value)
# normalized = normalized.unsqueeze(0).unsqueeze(0)

# print(normalized)

import torch

# Define your 9x9 matrix
matrix = torch.tensor([[1, 4, 7, 2, -2, 5, 8, 0, 3],
                       [2, 6, 0, 8, 1, 3, 5, 4, 7],
                       [3, 1, 7, 2, 4, 6, 0, 8, 5],
                       [2, 5, 1, 7, 4, 3, 6, 8, 0],
                       [0, 3, 6, 2, 5, 1, 8, 7, 4],
                       [8, 2, 0, 4, 6, 7, 1, 3, 5],
                       [4, 7, 3, 5, 0, 1, 8, 2, 6],
                       [6, 8, 5, 3, 7, 0, 2, 1, 4],
                       [1, 0, 4, 6, 2, 5, 7, 3, 8]])

# Determine the number of classes
num_classes = matrix.max() - matrix.min() + 1

# Create an empty one-hot encoded matrix
one_hot_matrix = torch.zeros((*matrix.shape, num_classes), dtype=torch.float)

# Fill in the one-hot encoded matrix
matrix = matrix + 2
for i in range(num_classes):    
    one_hot_matrix[..., i] = (matrix == i).float()

# Reshape the one-hot encoded matrix to match the input shape of the convolutional layer
reshaped_matrix = one_hot_matrix.permute(2, 0, 1).unsqueeze(0)

# Print the shape and values of the input tensor
print(reshaped_matrix.shape)
print(reshaped_matrix[0])