from Utils import *

if __name__ == '__main__':
    lines = read_lines("./lines")
    passenger_flows = read_transect_flows("./passenger_flows/workday/TransectVolume-30min/", "workday", lines)
    set_line_short_routes(lines, passenger_flows)
    read_arrival_rates("./passenger_flows/workday/ArrivalVolume-30min", lines, passenger_flows)
    read_alight_rates("./passenger_flows/workday/AlightVolume-30min", lines, passenger_flows)
    read_transfer_rates("./passenger_flows/workday/TransferRate", lines, passenger_flows)

    gen_timetables(100, lines, passenger_flows)

    sdasd = 0
