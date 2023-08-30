class Depot:
    def __init__(self, id):
        self.id = id
        self.conn_lines = []
        self.conn_stations = {}
        self.runtime = {}
        self.capacity = 0
