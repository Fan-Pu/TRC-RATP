from collections import defaultdict


class Depot:
    def __init__(self, id):
        self.id = id
        self.conn_lines = []
        self.conn_stations = defaultdict(list)  # key: line_id, value: list(station_id)
        self.runtime = {}
        self.capacity = 0
        self.maximum_flow = 0
