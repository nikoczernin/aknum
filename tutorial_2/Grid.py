# ===================================================
# Author: Nikolaus Czernin
# Script: Grid Class for 2D Gridworld Representation
# Description: A customizable 2D grid with optional hard borders,
#              item placement and retrieval, and visual display.
# ===================================================

import numpy as np

class Grid:
    def __init__(self, width, height, hard_borders=True, default_item="-"):
        self.width = width
        self.height = height
        self.default_item = default_item
        # self.grid = [[self.default_item(self) for x in range(width)] for y in range(height)]
        self.grid = np.empty((width, height), dtype=str)
        for y in range(height):
            for x in range(width):
                self.grid[x, y] = default_item
        # allow leaving borders (if allowed this typically results in death)
        self.hard_borders = hard_borders

    def put(self, item, x, y):
        self.grid[x, y] = item

    def get(self, x, y):
        return self.grid[x, y]

    def pop(self, x, y):
        self.grid[x, y] = self.default_item

    def __str__(self, *items, rulers=True):
        out = ""
        if rulers: out += "  " + "".join([f"  {i} " for i in range(self.width)]) + "\n"
        for y in range(self.height):
            if rulers: out += f"{y}   "
            for x in range(self.width):
                 out +=  str(self.get(x, y)) + "   "
            out += "\n"
        if rulers: out += "   "
        return out

    def draw(self, *args, **kwargs):
        print(self.__str__(*args, **kwargs))


if __name__ == "__main__":
    grid = Grid(10, 10)
    grid.put("G", 4, 5)
    grid.put("A", 0, 1)
    print(grid)