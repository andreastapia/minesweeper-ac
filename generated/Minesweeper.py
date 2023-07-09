import numpy as np

class MinesweeperEnvironment:
    def __init__(self, grid_size=9, num_mines=10):
        self.grid_size = grid_size
        self.num_mines = num_mines
        self.grid = None
        self.state = None
        self.is_game_over = None

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int)
        self.place_mines()
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int)
        self.is_game_over = False
        return self.state.copy()

    def place_mines(self):
        indices = np.random.choice(self.grid_size * self.grid_size, size=self.num_mines, replace=False)
        mine_locations = np.unravel_index(indices, (self.grid_size, self.grid_size))
        self.grid[mine_locations] = -1

    def get_state_size(self):
        return self.grid_size * self.grid_size

    def get_action_size(self):
        return self.grid_size * self.grid_size

    def step(self, action):
        row = action // self.grid_size
        col = action % self.grid_size

        if self.is_game_over or self.state[row, col] != 0:
            # Invalid move, penalize with a negative reward and end the episode
            return self.state.copy(), -10, True

        if self.grid[row, col] == -1:
            # Mine is uncovered, end the episode with a negative reward
            self.is_game_over = True
            self.state[row, col] = -1
            return self.state.copy(), -10, True

        self.uncover_cell(row, col)
        done = self.check_game_over()
        if done:
            return self.state.copy(), 10, True
        else:
            return self.state.copy(), 1, False

    def uncover_cell(self, row, col):
        if self.grid[row, col] == 0:
            self.state[row, col] = 1
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= row + i < self.grid_size and 0 <= col + j < self.grid_size:
                        if self.grid[row + i, col + j] == 0 and self.state[row + i, col + j] == 0:
                            self.uncover_cell(row + i, col + j)
                        else:
                            self.state[row + i, col + j] = 1
        else:
            self.state[row, col] = self.grid[row, col]

    def check_game_over(self):
        return np.all(self.state != 0)

    def render(self):
        for row in self.state:
            for cell in row:
                if cell == -1:
                    print("X", end=" ")
                elif cell == 0:
                    print("-", end=" ")
                else:
                    print(cell, end=" ")
            print()

    