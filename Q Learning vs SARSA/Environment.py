import tkinter as tk
import random

class GridWorld:
    def __init__(self, rows=10, cols=10):
        self.rows = rows
        self.cols = cols
        self.grid = [['W' for _ in range(cols)] for _ in range(rows)]
        self.track1_start = (0, 0)
        self.track1_end = (4, cols - 1)
        self.track2_start = (5, 0)
        self.track2_end = (9, cols - 1)
        self.track1_land = set()
        self.track2_land = set()
        self.generate_tracks()

    def generate_tracks(self):
        self.grid = [['W' for _ in range(self.cols)] for _ in range(self.rows)]

        # Place Start and End
        self.grid[self.track1_start[0]][self.track1_start[1]] = 'S1'
        self.grid[self.track1_end[0]][self.track1_end[1]] = 'E1'
        self.grid[self.track2_start[0]][self.track2_start[1]] = 'S2'
        self.grid[self.track2_end[0]][self.track2_end[1]] = 'E2'
        self.track1_land = {self.track1_start}
        self.track2_land = {self.track2_start}

        # Create a land corridor (mirror logic)
        corridor_rows_t1 = random.sample(range(0, 5), 3)
        corridor_rows_t2 = [r + 5 for r in corridor_rows_t1]  # mirror

        for col in range(1, self.cols - 1):
            for r1, r2 in zip(corridor_rows_t1, corridor_rows_t2):
                self.grid[r1][col] = 'L1'
                self.track1_land.add((r1, col))

                self.grid[r2][col] = 'L2'
                self.track2_land.add((r2, col))

        # Add land near the goal
        if self.grid[self.track1_end[0]][self.track1_end[1] - 1] == 'W':
            self.grid[self.track1_end[0]][self.track1_end[1] - 1] = 'L1'
            self.track1_land.add((self.track1_end[0], self.track1_end[1] - 1))

        if self.grid[self.track2_end[0]][self.track2_end[1] - 1] == 'W':
            self.grid[self.track2_end[0]][self.track2_end[1] - 1] = 'L2'
            self.track2_land.add((self.track2_end[0], self.track2_end[1] - 1))

    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols

    def is_terminal(self, pos):
        return pos == self.track1_end or pos == self.track2_end

    def get_state_type(self, pos):
        if not self.is_valid(pos):
            return 'W'
        return self.grid[pos[0]][pos[1]]

    def draw(self, canvas):
        block = 50
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.grid[i][j]
                color = {
                    'W': 'blue', 'L1': 'tan', 'L2': 'tan',
                    'S1': 'green', 'S2': 'green',
                    'E1': 'red', 'E2': 'red'
                }.get(val, 'white')
                canvas.create_rectangle(j * block, i * block, (j + 1) * block, (i + 1) * block, fill=color, outline='black')
