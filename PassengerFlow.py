class PassengerFlow:
    """
        class for the timetable (minutes, starts from 0)
    """

    def __init__(self, type_name, sectional_flows):
        self.type_name = type_name
        self.sectional_flows = sectional_flows
        self.d = {}  # d{l,s,t}, intensity of arriving passengers
        self.theta = {}  # theta{l,s,t}, intensity of alighting passengers
        self.phi = {}  # phi{l1,s1,l2,s2,t}, transfer rate
