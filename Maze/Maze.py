import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)  # Optional: makes maze reproducible

# Maze dimensions (odd numbers for walls + paths)
width, height = 8, 8
if width % 2 == 0:
    width += 1
if height % 2 == 0:
    height += 1

maze = np.ones((height, width), dtype=np.int8)

def carve_passages(x, y, depth=0, max_depth=10):
    # Bias: prefer right and down directions
    directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    random.shuffle(directions)
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 1 <= nx < width - 1 and 1 <= ny < height - 1:
            if maze[ny][nx] == 1:
                maze[ny][nx] = 0
                maze[y + dy // 2][x + dx // 2] = 0
                if depth < max_depth:
                    carve_passages(nx, ny, depth + 1, max_depth)

# Start maze generation
maze[1][1] = 0
carve_passages(1, 1, max_depth=6)  # Lower max_depth = simpler maze

# Add entrance and exit
maze[0][1] = 0                    # Entrance
maze[height - 1][width - 2] = 0   # Exit

# Display
plt.figure(figsize=(4, 4))
plt.imshow(maze, cmap='binary')
plt.axis('off')
plt.title('Simple 8x8 Maze')
plt.show()
