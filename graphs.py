import os
import matplotlib.pyplot as plt
import numpy as np


def running_average(x, N):
    cumsum = np.cumsum(np.insert(np.array(x), 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

figs = './figs/'
directory = './results/'

rewards = 'rewards/'
dir = directory+rewards
for filename in os.listdir(dir):
    if filename.endswith('.txt'):
        im_path = figs + filename.split('.')[0] + '.png'
        returns = np.loadtxt(dir + filename)        
        plt.figure(figsize=(15, 6))
        plt.plot(returns)
        plt.plot(running_average(returns, 1000))
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title(filename)
        plt.savefig(im_path)
        plt.show()

steps = 'steps/'
dir = directory+steps
for filename in os.listdir(dir):
    if filename.endswith('.txt'):
        stp = np.loadtxt(dir + filename)
        plt.figure(figsize=(15, 6))
        plt.plot(stp)
        plt.plot(running_average(stp, 1000))
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        plt.show()

avg1k = 'avg1k/'
dir = directory+avg1k
for filename in os.listdir(dir):
    if filename.endswith('.txt'):
        avg_1k = np.loadtxt(dir + filename)
        plt.figure(figsize=(15, 6))
        plt.plot(avg_1k)
        plt.plot(running_average(avg_1k, 1000))
        plt.xlabel("Episodes")
        plt.ylabel("Average reward in 1k steps")
        plt.show()