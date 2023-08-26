class TrainService:

    def __init__(self, id, direction, route):
        self.id = id
        self.direction = direction
        self.route = route
        self.arrs = []
        self.deps = []
        self.front_service = -1  # turnback connection
        self.next_service = -1
        self.first_service = -1
