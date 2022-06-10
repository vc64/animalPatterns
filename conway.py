"""
Conway's game of life simulation

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(7, 7))

shape = np.array([100, 100])

grid = np.zeros(shape+1)
ones = np.ones(shape+1)

# freq of 1s in the grid
replace_rate = 0.4

bool_mask = np.random.choice([0, 1], size=shape+1, p=((1 - replace_rate), replace_rate)).astype(bool)

grid[bool_mask] = ones[bool_mask]


def conway(row, col, grid):
    curr = grid[row, col]

    total = sum(grid[row-1:row+2, col+1]) + sum(grid[row-1:row+2, col]) + sum(grid[row-1:row+2, col-1]) - curr

    if curr == 1:
        if total == 2 or total == 3:
            return 1
    elif total == 3:
        return 1
    
    return 0



def update(frame_number, grid, img):
    new_grid = np.zeros(shape+1)
    for row in range(1, shape[0]):
        for col in range(1, shape[1]):
            new_grid[row][col] = conway(row, col, grid)
    
    grid[:] = new_grid[:]
    img.set_data(new_grid)

    return img,

# how fast the simulation updates (ms)
animation_rate = 50

img = plt.imshow(grid[1:-1, 1:-1], cmap='cividis', interpolation='nearest')
animation = FuncAnimation(fig, update, fargs = (grid, img, ),  interval=animation_rate, frames = 10, save_count = 50)
plt.show()