# ===================================================
# Author: Nikolaus Czernin
# Script: Grid Class for 2D Gridworld Representation
# Description: A customizable 2D grid with optional hard borders,
#              item placement and retrieval, and visual display.
# ===================================================

import numpy as np

class Grid:
    def __init__(self, h, w, hard_borders=True, default_item="-"):
        self.height = h
        self.width = w
        self.default_item = default_item
        self.grid = np.empty((h, w), dtype='<U10')
        self.max_item_len = len(default_item)

        for y in range(h):
            for x in range(w):
                self.grid[y, x] = default_item
        # allow leaving borders (if allowed this typically results in death)
        self.hard_borders = hard_borders

    def set_max_item_length(self):
        self.max_item_len = max(len(item) for item in self.grid.flat)

    def put(self, item, y, x):
        self.grid[y, x] = item
        self.set_max_item_length()

    def get(self, y, x):
        return self.grid[y, x]

    def pop(self, y, x):
        self.grid[y, x] = self.default_item
        self.set_max_item_length()

    def __str__(self, *items, rulers=True):
        out = ""
        cell_w = self.max_item_len
        out += "   " + "  ".join(str(i).rjust(self.max_item_len) for i in range(self.grid.shape[1])) + "\n"
        for y, row in enumerate(self.grid):
            out += f"{y}  " + "  ".join(str(item).rjust(self.max_item_len) for item in row) + "\n"
        return out

    def draw(self, *args, **kwargs):
        print(self.__str__(*args, **kwargs))

    @staticmethod
    def draw_grid(content:dict, round_to=None):
        # content should be a dict where the keys are the positions and
        # the values are the strings you want to print
        if round_to is not None:
            for key, value in content.items():
                if isinstance(value, float):
                    content[key] = round(value, round_to)
        # compute the max cell width and cap it with the round_to if given
        max_cell_width = max(len(str(v)) for v in content.values())
        out = ""
        y_prev = None # previous row number
        for (y, x), value in content.items():
            # turn the current value into a string
            value = str(value)
            # if the string is too short, pad it with whitespace
            while len(value) < max_cell_width: value = " " + value
            if y_prev is not None:
                if y > y_prev:
                    out += "\n"
            out += value + "  "
            # move the previous row number
            y_prev = y
        print(out)


if __name__ == "__main__":
    grid = Grid(10, 10)
    grid.put("G", 4, 5)
    grid.put("A", 0, 1)
    print(grid)