class Player(object):

    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def set_x(self, value):
        self.x = value
    def set_y(self, value):
        self.y = value
    def get_image(self):
        return self.image
    def get_direction(self):
        return self.direction
    def set_direction(self, value):
        self.direction = value
    def set_image(self, value):
        self.image = value
    def set_state(self, value):
        self.powered_up = value
    def get_state(self):
        return self.powered_up
    def at_intersection(self):
        horizontal_move = 0 in self.moves or 1 in self.moves
        vertical_move = 2 in self.moves or 3 in self.moves
        return (horizontal_move and vertical_move) or (self.get_direction() not in self.moves)

    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image
        self.direction = 0
        self.powered_up = False
        self.moves = set()
