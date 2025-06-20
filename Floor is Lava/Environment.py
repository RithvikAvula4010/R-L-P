import numpy as np
import random
import os

class FloorIsLavaEnv:
    def __init__(self, size=8, grid_file="saved_grid.npy"):
        self.size = size
        self.grid_file = grid_file
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (-2, 0),  # jump up
            5: (2, 0),   # jump down
            6: (0, -2),  # jump left
            7: (0, 2),   # jump right
        }

        # If saved grid exists, load it. Else, generate new and save it.
        if os.path.exists(self.grid_file):
            self.grid = np.load(self.grid_file)
        else:
            self.reset_grid()
            np.save(self.grid_file, self.grid)

        self.reset()

    def reset_grid(self):
        self.grid = np.full((self.size, self.size), 'L')

        # Start and Destination
        self.grid[0][0] = 'S'
        self.grid[self.size - 1][self.size - 1] = 'D'

        # Ensure destination has at least one land neighbor
        dest_neighbors = [(self.size - 2, self.size - 1), (self.size - 1, self.size - 2)]
        random.shuffle(dest_neighbors)
        for nx, ny in dest_neighbors:
            if 0 <= nx < self.size and 0 <= ny < self.size:
                self.grid[nx][ny] = 'G'
                break

        # Generate random land blocks
        land_blocks = random.randint(8, 15)
        for _ in range(land_blocks):
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            direction = random.choice(['H', 'V'])
            length = random.choice([1, 2])

            for i in range(length):
                xi, yi = x, y
                if direction == 'H':
                    yi = y + i
                else:
                    xi = x + i
                if 0 <= xi < self.size and 0 <= yi < self.size and self.grid[xi][yi] == 'L':
                    self.grid[xi][yi] = 'G'

    def reset(self):
        self.agent_pos = [0, 0]
        self.prev_pos = None
        self.visited = set()
        self.visited.add(tuple(self.agent_pos))
        return tuple(self.agent_pos)

    def is_valid(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def step(self, action):
        move = self.actions[action]
        new_x = self.agent_pos[0] + move[0]
        new_y = self.agent_pos[1] + move[1]

        reward = -1 if action <= 3 else -2
        done = False

        if self.is_valid(new_x, new_y):
            new_pos = (new_x, new_y)
            tile = self.grid[new_x][new_y]

            if new_pos == tuple(self.agent_pos):
                reward += -1  # penalty for not moving

            if tile == 'L':
                reward += -10
            elif tile in ['G', 'S']:
                if action > 3:
                    reward += 10
                if new_pos not in self.visited:
                    reward += 1
            elif tile == 'D':
                reward += 100
                done = True

            self.prev_pos = tuple(self.agent_pos)
            self.agent_pos = [new_x, new_y]
            self.visited.add(tuple(self.agent_pos))

        return tuple(self.agent_pos), reward, done

    def get_grid(self):
        return self.grid, tuple(self.agent_pos), self.visited
