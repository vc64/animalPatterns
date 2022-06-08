"""
===============
Rain simulation
===============

Simulates rain drops on a surface by animating the scale and opacity
of 50 scatter points.

Author: Nicolas P. Rougier
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(7, 7))

shape = np.array([100, 100])

grid = np.zeros(shape+1)
ones = np.ones(shape+1)

replace_rate = 0.3
bool_mask = np.random.choice([0, 1], size=shape+1, p=((1 - replace_rate), replace_rate)).astype(np.bool)

grid[bool_mask] = ones[bool_mask]

print(grid)


plt.imshow(grid, cmap='hot', interpolation='nearest')


def conway(row, col, grid):
    curr = grid[row, col]
    # print(grid[row-1:row+1][col+1])

    # if row == 0:

    # elif row == shape[1] - 1:
    #     total = 
    # if col == 0 or col == shape[0] - 1:


    total = sum(grid[row-1:row+1, col+1]) + sum(grid[row-1:row+1, col]) + sum(grid[row-1:row+1, col-1]) - curr
    if curr == 1:
        if total == 2 or total == 3:
            return 1
    elif total == 3:
        return 1
    
    return 0



def update(frame_number, grid):
    new_grid = grid.copy()
    for row in range(shape[0]):
        for col in range(shape[1]):
            new_grid[row][col] = conway(row, col, grid)
    
    # grid = np.random.randint(1, (100, 100))
    plt.imshow(new_grid[1:-1, 1:-1], cmap='hot', interpolation='nearest')
    grid[:] = new_grid[:]


# # Construct the animation, using the update function as the animation
# # director.
animation = FuncAnimation(fig, update, fargs = (grid, ),  interval=500)
plt.show()




# # Create new Figure and an Axes which fills it.
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_axes([0, 0, 1, 1], frameon=False)
# ax.set_xlim(0, 1), ax.set_xticks([])
# ax.set_ylim(0, 1), ax.set_yticks([])

# # Create rain data
# n_drops = 50
# rain_drops = np.zeros(n_drops, dtype=[('position', float, 2),
#                                       ('size',     float, 1),
#                                       ('growth',   float, 1),
#                                       ('color',    float, 4)])

# # Initialize the raindrops in random positions and with
# # random growth rates.
# rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))
# rain_drops['growth'] = np.random.uniform(50, 200, n_drops)

# # Construct the scatter which we will update during animation
# # as the raindrops develop.
# scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],
#                   s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
#                   facecolors='none')


# def update(frame_number):
#     # Get an index which we can use to re-spawn the oldest raindrop.
#     current_index = frame_number % n_drops

#     # Make all colors more transparent as time progresses.
#     rain_drops['color'][:, 3] -= 1.0/len(rain_drops)
#     rain_drops['color'][:, 3] = np.clip(rain_drops['color'][:, 3], 0, 1)

#     # Make all circles bigger.
#     rain_drops['size'] += rain_drops['growth']

#     # Pick a new position for oldest rain drop, resetting its size,
#     # color and growth factor.
#     rain_drops['position'][current_index] = np.random.uniform(0, 1, 2)
#     rain_drops['size'][current_index] = 5
#     rain_drops['color'][current_index] = (0, 0, 0, 1)
#     rain_drops['growth'][current_index] = np.random.uniform(50, 200)

#     # Update the scatter collection, with the new colors, sizes and positions.
#     scat.set_edgecolors(rain_drops['color'])
#     scat.set_sizes(rain_drops['size'])
#     scat.set_offsets(rain_drops['position'])


# # Construct the animation, using the update function as the animation
# # director.
# animation = FuncAnimation(fig, update, interval=10)
# plt.show()