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

np.seterr('raise')

# grid = np.zeros(shape+1)
# ones = np.ones(shape+1)

replace_rate = 0.1
# grid = np.random.choice([0, 1], size=shape+1, p=((1 - replace_rate), replace_rate))

rng = default_rng()
grid_active = np.around(rng.uniform(0.0, 1.0, shape+1), decimals = 5)
grid_inhib = np.around(rng.uniform(0.0, 1.0, shape+1), decimals = 5)

def diffuse(row, col, grid):
    curr = grid[row, col]

    # total = sum(grid[row-1:row+2, col+1]) + sum(grid[row-1:row+2, col]) + sum(grid[row-1:row+2, col-1]) - 1

    # attempt 1:
    # 0.05  0.2     0.05
    # 0.2   -1      0.2
    # 0.05  0.2     0.05

    # multiply freq of each surrounding square by corresponding rate


    # attempt 2:
    # 0.0625    0.1875  0.0625
    # 0.1875    -1      0.1875
    # 0.0625    0.1875  0.0625
    # uses proportion of circle present in edge vs corner squares in 3x3 grid
    # math will be included later


    edges = grid[row, col-1] + grid[row, col+1] + grid[row-1, col] + grid[row+1, col]
    corners = grid[row-1, col-1] + grid[row-1, col+1] + grid[row+1, col-1] + grid[row+1, col+1]
    # total = edges * 0.1875 + corners * 0.0625 - 1 * curr
    
    diffuse_ratio = 0.1 * np.array([0.1875, 0.0625, 1])
    
    total = edges * diffuse_ratio[0] + corners * diffuse_ratio[1] - curr * diffuse_ratio[2]



    return total


    


# def eaten():
#     # not sure how to do this


def changePeriodic(row, col, grid, rate):
    # return 0
    # curr = np.around(grid[row, col], decimals = 5)
    # # print(rate, curr)
    # try:
    #     return np.around(rate * curr * (1 - curr), decimals = 5)
    # except:
    #     print(curr)
    return rate * (0.9 - grid[row, col])

    # if rng.random(1) <= rate:
    #     return 0
    # return 0


prev = 1
def update(frame_num, gridA, gridI, img):
    new_grid_active = np.around(gridA, decimals = 5).copy()
    new_grid_inhib = np.around(gridI, decimals = 5).copy()
    # grid = gridA.copy()
    for row in range(1, shape[0]):
        for col in range(1, shape[1]):
            new_grid_active[row, col] += diffuse(row, col, gridA) + changePeriodic(row, col, gridA, 0.01)
            new_grid_inhib[row, col] += diffuse(row, col, gridI) - changePeriodic(row, col, gridI, 0.04)

            rxn = min(gridA[row, col], gridI[row, col] / 2)
            gridA[row, col] -= rxn
            gridI[row, col] += rxn * 1.5
    print(np.amax(new_grid_active))
    # if np.amax(new_grid_active) > prev:
    #     print(new_grid_active.reshape(shape+1).tolist())
    #     quit()

    gridA[:] = np.around(new_grid_active[:], decimals = 5)
    gridI[:] = np.around(new_grid_inhib[:], decimals = 5)

    # grid = new_grid_active * (new_grid_inhib ** 2)
    img.set_data(np.around(new_grid_active[1:-1, 1:-1]))

    return img,

animation_rate = 50

img = plt.imshow(grid_active[1:-1, 1:-1], cmap = "viridis", interpolation = "nearest")
animation = FuncAnimation(fig, update, fargs = (grid_active, grid_inhib, img,), interval = animation_rate, frames = 10)
plt.show()

