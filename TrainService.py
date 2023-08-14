class TrainService:

    def __init__(self, id, direction, route):
        self.id = id
        self.direction = direction
        self.route = route
        self.arrs = []
        self.deps = []
