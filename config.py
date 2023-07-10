""" Definition of constants values for Minesweeper project """

""" Quantity of episodes to test """
EPISODES = 30000

""" Default value for win and movements counters """
DEFAULT_VALUE =  0

""" Value when tile has a mine """
MINE = -1

""" Value when tile is closed """
CLOSED = -2

""" Value for maximum mines around tile """
MAX_MINES_AROUND = 8

""" Mine quantity for game """
NUM_MINES = 10

""" Width value for board """
WIDTH = 9

""" Height value for board """
HEIGHT = 9

""" Reward value for correct movement """
STEP_REWARD = 1

""" Reward value for winning the game """
WIN_REWARD = 10

""" Reward value for losing the game """
LOSE_REWARD = -10

""" Reward value for making a guess move """
GUESS_REWARD = -1

""" Reward value for repeated action """
REPEATED_STEP_REWARD = -1

""" Amount of features of the one-hot encoded state """
FEATURES = 11