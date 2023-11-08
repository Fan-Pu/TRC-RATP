from Utils import *

if __name__ == '__main__':
    lines = read_lines("./lines")
    depots = read_depots("./depots", lines)
    passenger_flows = read_transect_flows("./passenger_flows/workday/TransectVolume-30min/", "workday", lines)
    set_line_short_routes(lines, passenger_flows)
    read_arrival_rates("./passenger_flows/workday/ArrivalVolume-30min", lines, passenger_flows)
    read_alight_rates("./passenger_flows/workday/AlightVolume-30min", lines, passenger_flows)
    read_transfer_rates("./passenger_flows/workday/TransferRate", lines, passenger_flows)

    solution_pool, constant_string, solution_summary = gen_timetables(100, lines, depots, passenger_flows)

    root_folder = f"./results/"
    if os.path.exists(root_folder):
        shutil.rmtree(root_folder)
        os.mkdir(root_folder)
    else:
        os.mkdir(root_folder)

    for i in range(0, len(solution_pool)):
        solution_folder = root_folder + f"{i}/"
        os.mkdir(solution_folder)
        solution = solution_pool[i]
        export_timetable(solution_folder, solution['timetable_net'], lines)
        export_allocation_plan(solution_folder, solution)
        export_solution_summary(solution_folder, solution_summary)

    export_constants(root_folder, constant_string)

    sdasd = 0
