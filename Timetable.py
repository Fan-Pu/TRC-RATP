class Timetable:
    """
    class for the timetable
    """

    def __init__(self, line_id):
        self.up_services = []  # up direction train services
        self.dn_services = []  # dn direction train services
        self.line_id = line_id  # the line index of this timetable
