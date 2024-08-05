class Ghost():
    def get_x(self):
        return self.x
    def get_name(self):
        return self.name
    def get_image(self):
        return self.image
    def get_y(self):
        return self.y

    def set_x(self, x):
        self.x = x
    def set_y(self, y):
        self.y = y
    def set_direction(self, value):
        self.direction = value
    def get_direction(self):
        return self.direction
    def at_intersection(self):
        return len(self.moves) > 2

    def __init__(self, x, y, name, image):
        self.x = x
        self.y = y
        self.name = name
        self.image = image
        self.direction = -1
        self.out_of_box = False
        self.moves = set()
        self.target_direction = None
        self.direction_cooldown = 0
        self.direction_cooldown_time = 2
        self.persistence_time = 7
        self.dead = False
        self.in_corner = False
        self.going_to_corner = False



