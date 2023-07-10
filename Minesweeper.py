import config
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from random import randint
import torch

class MinesweeperDiscrete(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self):
        """
        Returns a valid action action

        Parameters
        ----
        config: 
        """
        self.COUNTERMEM = config.DEFAULT_VALUE
        self.COUNTERWINS = config.DEFAULT_VALUE
        self.num_features = config.FEATURES
        self.mine = config.MINE
        self.closed = config.CLOSED
        self.max_mines_around = config.MAX_MINES_AROUND
        self.num_mines = config.NUM_MINES
        self.width = config.WIDTH
        self.height = config.HEIGHT
        self.step_reward = config.STEP_REWARD
        self.repeated_step_reward = config.REPEATED_STEP_REWARD
        self.win_reward = config.WIN_REWARD
        self.lose_reward = config.LOSE_REWARD
        self.guess_step_reward = config.GUESS_REWARD
        self.num_actions = 0
        self.num_clicks = 0
        self.mines_board = self.place_mines(self.height, self.width, self.num_mines)
        self.showed_board = np.ones((self.height, self.width), dtype=int) * self.closed
        self.conv_input_board = np.zeros((1,1,self.height, self.width))
        self.observation_space = spaces.Box(low=self.closed, high=self.max_mines_around,
                                            shape=(self.height, self.width), dtype=np.int8)
        self.action_space = spaces.Discrete(self.height * self.width)

    def is_new_move(self, showed_board, x, y):
        """ return true if this is not an already clicked place"""
        return showed_board[x, y] == self.closed


    def is_valid(self, x, y):
        """ returns if the coordinate is valid"""
        return (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)


    def is_win(self, showed_board):
        """ return if the game is won """
        return np.count_nonzero(showed_board == self.closed) == self.num_mines


    def is_mine(self, mines_board, x, y):
        """return if the coordinate has a mine or not"""
        return mines_board[x, y] == self.mine

    def check_guess(self, showed_board, x, y):
        sum = 0
        valid = 0
        for _x in range(x - 1, x + 2):
            for _y in range(y - 1, y + 2):
                if self.is_valid(_x, _y) and (x,y) != (_x,_y):
                    valid += 1
                    sum += showed_board[_x, _y]

        if sum == self.closed * valid:
            return True
        
        return False

    def place_mines(self, height, width, num_mines):
        """generate a board, place mines randomly"""
        mines_placed = 0
        mines_board = np.zeros((height, width), dtype=int)
        while mines_placed < num_mines:
            rnd = randint(0, height * width)
            x = int(rnd / width)
            y = int(rnd % height)
            if self.is_valid(x, y):
                if not self.is_mine(mines_board, x, y):
                    mines_board[x, y] = self.mine
                    mines_placed += 1
        return mines_board
    
    def count_neighbour_mines(self, x, y):
        """return number of mines in neighbour cells given an x-y coordinate

            Cell -->Current Cell(row, col)
            N -->  North(row - 1, col)
            S -->  South(row + 1, col)
            E -->  East(row, col + 1)
            W -->  West(row, col - 1)
            N.E --> North - East(row - 1, col + 1)
            N.W --> North - West(row - 1, col - 1)
            S.E --> South - East(row + 1, col + 1)
            S.W --> South - West(row + 1, col - 1)
        """
        neighbour_mines = 0
        for _x in range(x - 1, x + 2):
            for _y in range(y - 1, y + 2):
                if self.is_valid(_x, _y):
                    if self.is_mine(self.mines_board, _x, _y):
                        neighbour_mines += 1
        return neighbour_mines

    def open_neighbour_cells(self, showed_board, x, y):
        """return number of mines in neighbour cells given an x-y coordinate

            Cell -->Current Cell(row, col)
            N -->  North(row - 1, col)
            S -->  South(row + 1, col)
            E -->  East(row, col + 1)
            W -->  West(row, col - 1)
            N.E --> North - East(row - 1, col + 1)
            N.W --> North - West(row - 1, col - 1)
            S.E --> South - East(row + 1, col + 1)
            S.W --> South - West(row + 1, col - 1)
        """
        for _x in range(x-1, x+2):
            for _y in range(y-1, y+2):
                if self.is_valid(_x, _y):
                    if self.is_new_move(showed_board, _x, _y):
                        showed_board[_x, _y] = self.count_neighbour_mines(_x, _y)
                        if showed_board[_x, _y] == 0:
                            showed_board = self.open_neighbour_cells(showed_board, _x, _y)
        return showed_board

    def get_next_state(self, state, x, y):
        """
        Get the next state.

        Parameters
        ----
        state : (np.array)   visible board
        x : int    location
        y : int    location

        Returns
        ----
        next_state : (np.array)    next visible board
        game_over : (bool) true if game over

        """
        showed_board = state
        game_over = False

        if self.is_mine(self.mines_board, x, y) and self.num_clicks == 0:
            reshaped_mine_boards = self.mines_board.reshape(1, (self.width * self.height))
            non_bomb_movement = np.random.choice(np.nonzero(reshaped_mine_boards != self.mine)[1])
            x = int(non_bomb_movement / self.width)
            y = int(non_bomb_movement % self.height)

        if self.is_mine(self.mines_board, x, y):
            showed_board[x, y] = self.mine
            game_over = True
        else:
            showed_board[x, y] = self.count_neighbour_mines(x, y)
            if showed_board[x, y] == 0:
                showed_board = self.open_neighbour_cells(showed_board, x, y)
        self.showed_board = showed_board

        self.num_clicks += 1
        return showed_board, game_over
    
    def get_onehot_encode_board(self, showed_board):
        showed_board = torch.tensor(showed_board)
        min_value = self.closed
        max_value = self.max_mines_around
        num_classes = max_value - min_value + 1

        showed_board = showed_board + 2 #para que la matrix parta de 0 a 11 en la iteración
        one_hot_matrix = torch.zeros((*showed_board.shape, num_classes), dtype=torch.float)
        for i in range(num_classes):
            one_hot_matrix[..., i] = (showed_board == i).float()
        
        reshaped_matrix = one_hot_matrix.permute(2, 0, 1).unsqueeze(0)

        return reshaped_matrix
    
    def get_conv_input(self, showed_board):
        showed_board = torch.tensor(showed_board)
        min_value = self.closed
        max_value = self.max_mines_around
        normalized = (showed_board - min_value) / (max_value - min_value)
        return normalized.unsqueeze(0).unsqueeze(0)

    def reset(self, seed=None, options=None):
        """
        Reset a new game episode. See gym.Env.reset()

        Returns
        ----
        next_state : (np.array, int)    next board
        """
        self.mines_board = self.place_mines(self.height, self.width, self.num_mines)
        self.showed_board = np.ones((self.height, self.width), dtype=int) * self.closed
        self.num_actions = 0
        self.num_clicks = 0
        self.conv_input_board = self.get_onehot_encode_board(self.showed_board)
        return self.conv_input_board, {}

    def step(self, action):
        """
        See gym.Env.step().

        Parameters
        ----
        action : np.array    location

        Returns
        ----
        next_state : (np.array)    next board
        reward : float        the reward for action
        done : bool           whether the game end or not
        info : {}
        """
        state = self.showed_board

        #eg. accion 10 en tablero de 9x9
        #x = 10/9 = 1 int
        #y = 10 % 9 = 1

        #caso accion 27
        #x = 27 / 9 = 3
        #y = 27 % 9 = 0
        x = int(action / self.width)
        y = int(action % self.height)
        
        next_state, reward, done, info = self.next_step(state, x, y)
        self.showed_board = next_state
        self.num_actions += 1
        next_state_conv_input = self.get_onehot_encode_board(next_state)
        self.conv_input_board = next_state_conv_input

        info['valid_actions'] = (next_state.flatten() == self.closed)
        info['num_actions'] = self.num_actions
        truncated = False
        
        return next_state_conv_input, reward, done, truncated, info

    def next_step(self, state, x, y):
        """
        Get the next observation, reward, done, and info.

        Parameters
        ----
        state : (np.array)    visible board
        x : int    location
        y : int    location

        Returns
        ----
        next_state : (np.array)    next visible board
        reward : float               the reward
        done : bool           whether the game end or not
        info : {}
        """

        showed_board = state
        if not self.is_new_move(showed_board, x, y):
            return showed_board, self.repeated_step_reward, False, {}
        
        while True:
            state, game_over = self.get_next_state(showed_board, x, y)

            if self.check_guess(showed_board, x, y):
                return state, self.guess_step_reward, False, {}

            if not game_over:
                if self.is_win(state):
                    print("HORRAAAAAY!")
                    return state, self.win_reward, True, {}
                else:
                    return state, self.step_reward, False, {}
            else:
                #print("BOOOOM!")
                return state, self.lose_reward, True, {}
    
    def render(self, mode='human'):
        print(self.showed_board)