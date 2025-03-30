from Item import Item
import numpy as np


class Grid:
    def __init__(self, width, height, hard_borders=True, default_item=Item):
        self.width = width
        self.height = height
        self.default_item = default_item
        # self.grid = [[self.default_item(self) for x in range(width)] for y in range(height)]
        self.grid = np.empty((width, height), dtype=object)
        for y in range(height):
            for x in range(width):
                self.grid[x, y] = default_item(self, x, y)
        # allow leaving borders (if allowed this typically results in death)
        self.hard_borders = hard_borders

    def put(self, item, x, y):
        self.grid[x, y] = item

    def get(self, x, y):
        return self.grid[x, y]

    def pop(self, x, y):
        self.grid[x, y] = self.default_item(self, x, y)

    def may_i_move_here(self, x, y):
        if self.hard_borders:
            if  self.is_this_out_of_bounds(x, y):
                return False
        # if the field is occupied by anything blocking the way, you cant move here
        if self.get(x, y).blocks_field:
            return False
        return True

    def is_this_move_possible(self, x1, y1, delta_x, delta_y):
        # starting position: (x1, x2)
        destination = (x1 + delta_x, y1 + delta_y)
        # check if you can move there
        return self.may_i_move_here(*destination)

    def is_this_out_of_bounds(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return False

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



