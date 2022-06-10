"""
reaction diffusion simulation
"""

# bunnies and food analogy (bunnies are inhibitors and food are activators)

# food' = food + (diffused food - eaten food + new food)
# bunnies' = bunnies + (diffused bunnies + new bunnies - dead bunnies)

# diffused calculation based on average concentration surrounding tile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(7, 7))

shape = np.array([100, 100])

grid = np.zeros(shape+1)
ones = np.ones(shape+1)

def diffuse(row, col, grid):
    curr = grid[row, col]

    # total = sum(grid[row-1:row+2, col+1]) + sum(grid[row-1:row+2, col]) + sum(grid[row-1:row+2, col-1]) - 1

    # attempt 1:
    # 0.05  0.2     0.05
    # 0.2   0.05    0.2
    # 0.05  0.2     0.05

    # multiply freq of each surrounding square by corresponding rate

    edges = grid[row, col-1] + grid[row, col+1] + grid[row-1, col] + grid[row+1, col]
    corners = grid[row-1, col-1] + grid[row-1, col+1] + grid[row-1, col+1] + grid[row+1, col+1]
    total = edges * 0.2 + corners * 0.05 - 1

    return total




def update(frame_num, grid, img):
    new_grid = grid.copy()
    for row in range(1, shape[0]):
        for col in range(1, shape[1]):
            new_grid[row, col] += diffuse(row, col, grid) - eaten() + grown()

