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
from scipy.ndimage.filters import gaussian_filter

from numba import jit

# grid_active = rng.uniform(0.0, 1.0, shape+1)
# grid_inhib = rng.uniform(0.0, 1.0, shape+1)

@jit(nopython=True)
def diffuse(row, col, grid):
    curr = grid[row, col]

    # total = sum(grid[row-1:row+2, col+1]) + sum(grid[row-1:row+2, col]) + sum(grid[row-1:row+2, col-1]) - 1

    ## attempt 1:
    ## 0.05  0.2     0.05
    ## 0.2   -1      0.2
    ## 0.05  0.2     0.05

    ## multiply freq of each surrounding square by corresponding rate


    ## attempt 2:
    ## 0.0625    0.1875  0.0625
    ## 0.1875    -1      0.1875
    ## 0.0625    0.1875  0.0625
    ## uses proportion of circle present in edge vs corner squares in 3x3 grid
    ## math will be included later


    edges = grid[row, col-1] + grid[row, col+1] + grid[row-1, col] + grid[row+1, col]
    corners = grid[row-1, col-1] + grid[row-1, col+1] + grid[row+1, col-1] + grid[row+1, col+1]
    
    # diffuse_ratio = 1 * np.array([0.1875, 0.0625, 1])
    diffuse_ratio = np.array([0.2, 0.05, 1])
    
    total = edges * diffuse_ratio[0] + corners * diffuse_ratio[1] - curr * diffuse_ratio[2]

    ## attempt 3: 
    ## how can i make diffusion rate depend on concentration of nearby tiles?

    ## base rates will be the same as attempt 2, but only when all concentrations are equal in the 8 squares around
    ## need to vary all rates based on current concentration
    
    # # for i in range(3):
    # #     for j in range(3):
            

    #         grid[row+(i-1), col+(j-1)]

    # edges = [grid[row, col-1], grid[row, col+1], grid[row-1, col], grid[row+1, col]]
    # corners = [grid[row-1, col-1], grid[row-1, col+1], grid[row+1, col-1], grid[row+1, col+1]]

    # total = 0
    # diffuse_ratio = np.array([0.1875, 0.0625, 1])

    # for edge in edges:
    #     total += (edge - curr) * diffuse_ratio[0]

    # for corn in corners:
    #     total += (corn - curr) * diffuse_ratio[1]

    # total -= curr
    
    # if curr < 0:
    #     total *= max((curr-1), 0)
    # else:
    #     total *= max((1-curr), 0)
    
    # total *= max((1-curr), 0)
    # print(total)

    return total


    


# def eaten():
#     # not sure how to do this


# def changePeriodic(row, col, grid, rate):
#     # return 0
#     # curr = np.around(grid[row, col], decimals = 5)
#     # # print(rate, curr)
#     # try:
#     #     return np.around(rate * curr * (1 - curr), decimals = 5)
#     # except:
#     #     print(curr)
#     return rate * (1 - grid[row, col])

#     # if rng.random(1) <= rate:
#     #     return 0
#     # return 0

@jit(nopython=True)
def update(gridA, gridI):
    new_grid_active = gridA.copy()
    new_grid_inhib = gridI.copy()
    # grid = gridA.copy()
    for row in range(1, shape[0]):
        for col in range(1, shape[1]):
            # rxn = min(gridA[row, col], gridI[row, col] / 2)
            # new_grid_active[row, col] -= rxn
            # new_grid_inhib[row, col] += rxn

            # rxn = min(1, gridA[row, col]) * min(1, gridI[row, col]) ** 2
            rxn = gridA[row, col] * gridI[row, col] ** 2
            # new_grid_active[row, col] -= rxn
            # new_grid_inhib[row, col] += rxn

            f = 0.0367
            k = 0.0649
            

            new_grid_active[row, col] += diffuse(row, col, gridA) - rxn + (1 - gridA[row, col]) * f
            new_grid_inhib[row, col] += 0.5 * diffuse(row, col, gridI) + rxn - gridI[row, col] * (f + k)

            # rxn = min(gridA[row, col], gridI[row, col] / 2)
            # gridA[row, col] -= rxn
            # gridI[row, col] += rxn * 1.5
    # print(np.amax(new_grid_active), np.amin(new_grid_active))
    # print(new_grid_active)
    # if np.amax(new_grid_active) > prev:
    #     # print(new_grid_active.reshape(shape+1).tolist())
    #     q = True

    gridA[:] = new_grid_active[:]
    gridI[:] = new_grid_inhib[:]

    # grid = new_grid_active * (new_grid_inhib ** 2)

    grid = gridA - gridI
    # print(gridA)

    # if np.amax(new_grid_active) > prev:
    #     # print(new_grid_active.reshape(shape+1).tolist())
    #     quit()

    
    # img.set_data(gridA[1:-1, 1:-1])

    # blurred = gaussian_filter(gridA, sigma=1)

    # img.set_data(blurred[1:-1, 1:-1])

    return [gridA, gridI, grid]


def updateN(frame_num, gridA, gridI, img, N):
    for x in range(N):
        gridA, gridI, out = update(gridA, gridI)
    
    img.set_data(out)
    return img,




fig = plt.figure(figsize=(7, 7))

shape = np.array([150, 150])

np.seterr('raise')

# grid = np.zeros(shape+1)
# ones = np.ones(shape+1)

# replace_rate = 0.5
# grid = np.random.choice([0, 1], size=shape+1, p=((1 - replace_rate), replace_rate))

# rng = default_rng()
grid_active = np.ones(shape+1)
grid_inhib = np.zeros(shape+1)

x = int(shape[0] / 2)
y = int(shape[1] / 2)
grid_inhib[x-1:x+1, y-1:y+1] += 1


animation_rate = 10

img = plt.imshow(grid_inhib[1:-1, 1:-1], cmap = "viridis", interpolation = "nearest")
animation = FuncAnimation(fig, updateN, fargs = (grid_active, grid_inhib, img, 50,), 
                            interval = animation_rate, blit=True)
plt.show()


