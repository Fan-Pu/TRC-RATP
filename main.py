from Utils import *

if __name__ == '__main__':
    lines = read_lines("./lines")
    passenger_flows = read_passenger_flows("./passenger_flows/workday", "workday", lines)
    set_line_short_routes(lines, passenger_flows)
    sdasd = 0
