from tutorial_2.GridWorld import GridWorld

class WindyGridWorld(GridWorld):
    def __init__(self, w, h, forces):
        super().__init__(self, w, h)
        # forces should be an array of length w of integer values
        self.forces = forces

