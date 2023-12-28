import shutil

from Utils import *

if __name__ == '__main__':
    lines = read_lines("./lines")
    depots = read_depots("./depots", lines)

    passenger_flows = read_transect_flows("./passenger_flows/workday/TransectVolume-30min/", "workday", lines)
    set_line_short_routes(lines, passenger_flows)
    read_arrival_rates("./passenger_flows/workday/ArrivalVolume-30min", lines, passenger_flows)
    read_alight_rates("./passenger_flows/workday/AlightVolume-30min", lines, passenger_flows)
    read_transfer_rates("./passenger_flows/workday/TransferRate", lines, passenger_flows)

    # passenger_flows = read_transect_flows("./passenger_flows/holiday/TransectVolume-30min/", "holiday", lines)
    # set_line_short_routes(lines, passenger_flows)
    # read_arrival_rates("./passenger_flows/holiday/ArrivalVolume-30min", lines, passenger_flows)
    # read_alight_rates("./passenger_flows/holiday/AlightVolume-30min", lines, passenger_flows)
    # read_transfer_rates("./passenger_flows/holiday/TransferRate", lines, passenger_flows)

    # passenger_flows = read_transect_flows("./passenger_flows/weekend/TransectVolume-30min/", "weekend", lines)
    # set_line_short_routes(lines, passenger_flows)
    # read_arrival_rates("./passenger_flows/weekend/ArrivalVolume-30min", lines, passenger_flows)
    # read_alight_rates("./passenger_flows/weekend/AlightVolume-30min", lines, passenger_flows)
    # read_transfer_rates("./passenger_flows/weekend/TransferRate", lines, passenger_flows)

    random.seed(2023)
    incumbent_solution, constant_string, solution_summary = execute_algorithm(lines, depots, passenger_flows)

    root_folder = f"./results/"
    if os.path.exists(root_folder):
        shutil.rmtree(root_folder)
        os.mkdir(root_folder)
    else:
        os.mkdir(root_folder)

    solution_folder = root_folder + "best_solution/"
    os.mkdir(solution_folder)
    export_timetable(solution_folder, incumbent_solution['timetable_net'], lines)
    export_allocation_plan(solution_folder, incumbent_solution)
    export_solution_summary(solution_folder, solution_summary)

    export_constants(root_folder, constant_string)

    sdasd = 0
