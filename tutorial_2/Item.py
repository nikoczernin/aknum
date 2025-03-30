# class IllegalMoveError(Exception):
#     def __init__(self, reason=None):
#         msg = "Illegal move"
#         if reason:
#             msg += f" – {reason}"
#         super().__init__(msg)


class Item:
    def __init__(self, grid, x, y, label="-", blocks_path=False):
        self.grid = grid
        self.grid.put(self, x, y) # upon spawning put the item onto a grid
        self.blocks_field = blocks_path # obstacles will block any agent from entering this Item's field
        self.position = (x, y)
        self.label = label # the label gets returned when printing the grid
        if not isinstance(blocks_path, bool): raise Exception("blocks_path must be a boolean")
        self.blocks_path = blocks_path # does the item occupy the space or can it be stepped on?

    def __str__(self):
        return self.label

    # set the item position on its grid
    def put(self, x, y):
        # adjust the item position
        self.position = (x, y)
        # put the item back on the grid
        self.grid.put(self, x, y)


class Agent(Item):
    # inherit from Item
    def __init__(self, grid, x, y, label):
        super().__init__(grid, x, y, label, blocks_path=True)

    # move the agent on his grid
    def move(self, x=0, y=0):
        # args: x, y are the position deltas as ints
        # returns: None
        # moving off bounds is handled by the grid itself,
        # the grid either allows moving off bounds or it doesnt
        # if moving off-bounds means termination, that must be handled by the environment
        # first get the starting position
        x0, y0 = self.position
        # see if the agent can even land in his destination
        # it may be obstructed by obstacles or the grid boundaries
        if self.grid.may_i_move_here(x0 + x, y0 + y):
            # remove the item from its current position in its grid
            self.grid.pop(x0, y0)
            # put it in the new position
            self.put(x0 + x, y0 + y)

    # is it possible to this agent by x and y
    def can_i_make_this_move(self, x_delta, y_delta):
        # args: x, y are the position deltas as ints
        # returns: boolean
        return self.grid.may_i_move_here(self.position[0] + x_delta, self.position[1] + y_delta)

class Flag(Item):
    def __init__(self, grid, x, y, label):
        super().__init__(grid, x, y, label, blocks_path=False)
