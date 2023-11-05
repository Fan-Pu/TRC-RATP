from collections import defaultdict


class Depot:
    def __init__(self, id):
        self.id = id
        self.conn_lines = []
        self.conn_stations = defaultdict(list)  # key: line_id, value: list(station_id)
        self.runtime = {}  # key (line_id,station_id)
        self.capacity = 0
        self.maximum_flow = 0
        self.f_values = {}  # key: line_id
