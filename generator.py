"""
reaction diffusion simulation
"""

# bunnies and food analogy (bunnies are inhibitors and food are activators)

# food' = food + (diffused food - eaten food + new food)
# bunnies' = bunnies + (diffused bunnies + new bunnies - dead bunnies)

# diffused calculation based on average concentration surrounding tile

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(7, 7))

shape = np.array([100, 100])

# grid = np.zeros(shape+1)
# ones = np.ones(shape+1)

replace_rate = 0.1
# grid = np.random.choice([0, 1], size=shape+1, p=((1 - replace_rate), replace_rate))

rng = default_rng()
grid = rng.uniform(0.0, 1.0, shape+1)

def diffuse(row, col, grid):
    curr = grid[row, col]

    # total = sum(grid[row-1:row+2, col+1]) + sum(grid[row-1:row+2, col]) + sum(grid[row-1:row+2, col-1]) - 1

    # attempt 1:
    # 0.05  0.2     0.05
    # 0.2   -1      0.2
    # 0.05  0.2     0.05

    # multiply freq of each surrounding square by corresponding rate

    edges = grid[row, col-1] + grid[row, col+1] + grid[row-1, col] + grid[row+1, col]
    corners = grid[row-1, col-1] + grid[row-1, col+1] + grid[row+1, col-1] + grid[row+1, col+1]
    total = edges * 0.1875 + corners * 0.0625 - 1 * curr

    return total


    # attempt 2:
    # 0.0625    0.1875  0.0625
    # 0.1875    -1      0.1875
    # 0.0625    0.1875  0.0625
    # uses proportion of circle present in edge vs corner squares in 3x3 grid
    # math will be included later


# def eaten():
#     # not sure how to do this


# def grown():
#     grow_rate = 0.02


def update(frame_num, grid, img):
    new_grid = grid.copy()
    for row in range(1, shape[0]):
        for col in range(1, shape[1]):
            new_grid[row, col] += diffuse(row, col, grid)
    
    # print(new_grid[45, 56])
    grid[:] = new_grid[:]
    img.set_data(new_grid[1:-1, 1:-1])

    return img,

animation_rate = 50

img = plt.imshow(grid[1:-1, 1:-1], cmap = "inferno", interpolation = "nearest")
animation = FuncAnimation(fig, update, fargs = (grid, img,), interval = animation_rate, frames = 10)
plt.show()

