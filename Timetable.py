class Timetable:
    """
    class for the timetable
    """

    def __init__(self, line_id):
        self.services = {}
        self.up_services = []
        self.dn_services = []
        self.line_id = line_id  # the line index of this timetable

        # !!! any time slots in this file refers to seconds between [START_TIME*60, END_TIME*60]

        self.p_wait = {}  # waiting passengers, key: sta_id, staring from START_TIME
        self.p_train = {}  # en-route passengers when it leaves the station, key: service_id
        self.p_alight = {}  # alight passengers, key: service_id
        self.p_transfer = {}  # transfer passengers, key: service_id
