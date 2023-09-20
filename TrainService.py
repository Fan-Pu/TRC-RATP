class TrainService:

    def __init__(self, id, direction, route, line_id):
        self.id = id
        self.direction = direction
        self.route = route
        self.line_id = line_id
        self.arrs = []
        self.deps = []
        self.front_service = -1  # turnback connection
        self.next_service = -1
        self.first_service = -1
        self.last_service = -1