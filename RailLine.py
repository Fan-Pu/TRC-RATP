class RailLine:

    def __init__(self, row):
        self.line_name = row[0]
        self.line_id = row[1]
        self.routes = []  # available train routes
        self.up_stations = []  # up direction stations
        self.dn_stations = []  # down direction stations
        self.runtimes = {}  # section runtimes
        self.dwells = {}  # station dwell times
        self.conn_depots = []  # indices of depots that connect with this line
        self.stations_conn_depots = {}  # stations and their connect depots
        self.headway_min = row[2]
        self.headway_max = row[3]
        self.turnback_min = row[4]
        self.turnback_max = row[5]
