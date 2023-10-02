class RailLine:

    def __init__(self, row):
        self.line_name = row[0]
        self.line_id = row[1]
        self.routes = []  # available train routes
        self.up_routes = []
        self.dn_routes = []
        self.is_loop_line = False
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
        self.platform_names = {}
        self.transfer_pairs = []  # indicate transfer connections, key (l1,s1,l2,s2)

    def get_runtime(self, start_station, end_station):
        # from arrive time of start_station to arrive time of end_station
        runtime = 0
        if end_station <= start_station:
            raise ValueError("end_station must be greater than start_station")
        for i in range(start_station, end_station):
            runtime += self.dwells[i] + self.runtimes[i, i + 1]
        return runtime
