class Timetable:
    """
    class for the timetable
    """

    def __init__(self, line_id):
        self.services = {}
        self.up_services = []
        self.dn_services = []
        self.line_id = line_id  # the line index of this timetable
        self.sum_wait_time = 0  # the sum of passenger waiting time
        self.sum_wait_time_max = 0  # the upper bound of sum_wait_time
        self.sum_wait_time_min = 0  # the lower bound of sum_wait_time

        # !!! any time slots in this file refers to seconds between [START_TIME * 60, END_TIME * 60]

        self.p_wait = {}  # waiting passengers, key: sta_id, staring from START_TIME
        self.p_train = {}  # en-route passengers when it leaves the station, key: service_id
        # self.p_alight = {}  # alight passengers, key: service_id
        # self.p_transfer = {}  # transfer passengers, key: service_id
