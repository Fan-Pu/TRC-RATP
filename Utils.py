import math
import os
import csv
import statistics
from RailLine import *
from PassengerFlow import *
import pandas as pd

INTERVAL = 30
START_TIME = 5 * 60
END_TIME = 24 * 60
PEAK_HOURS = [(7 * 60, 9 * 60), (17 * 60, 19 * 60)]
FLOW_FACTOR = 1.5


def read_lines(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv') and file != 'general_info.csv']

    if not csv_files:
        print("No CSV files found in the given folder.")
        return

    lines = {}
    # read general_info.csv
    csv_file_path = os.path.join(folder_path, 'general_info.csv')
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            rail_line = RailLine(row)
            lines[rail_line.line_id] = rail_line

    # read other files
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        line_id = csv_file.split(".")[0].split("-")[2]
        with open(csv_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)
            rail_line = lines[line_id]
            for row in reader:
                station_name, station_id, dwell_time, runtime, is_transfer_station, conn_depot_id = row

                rail_line.dwells[station_id] = int(dwell_time)
                if not int(station_id) in rail_line.up_stations:
                    rail_line.up_stations.append(int(station_id))

                    if conn_depot_id != '':
                        rail_line.conn_depots.append(int(conn_depot_id))
                        rail_line.stations_conn_depots[int(station_id)] = int(conn_depot_id)

                    # add sections
                    if runtime != '':
                        rail_line.runtimes[(int(station_id), int(station_id) + 1)] = int(runtime)
                else:
                    station_cnt = len(rail_line.up_stations)
                    temp_runtime = rail_line.runtimes[(station_cnt - 1, station_cnt)]
                    del rail_line.runtimes[(station_cnt - 1, station_cnt)]
                    rail_line.runtimes[(station_cnt - 1, 0)] = temp_runtime

                # add up direction platform names
                if (station_id, 0) not in rail_line.platform_names:
                    rail_line.platform_names[(int(station_id), 0)] = station_name

    # generate values for down direction
    for key, line in lines.items():
        # generate stations
        line.dn_stations = [i for i in range(len(line.up_stations), 2 * len(line.up_stations))]
        runtimes = {}
        station_cnt = len(line.up_stations)
        for (i, j), runtime in line.runtimes.items():
            runtimes[(2 * station_cnt - 1 - j, 2 * station_cnt - 1 - i)] = runtime
        line.runtimes.update(runtimes)
        # update stations_conn_depots
        stations_conn_depots = {}
        for sta_id, depot_id in line.stations_conn_depots.items():
            stations_conn_depots[2 * len(line.up_stations) - 1 - sta_id] = depot_id
        line.stations_conn_depots.update(stations_conn_depots)
        # add down direction platform names
        for station_id in line.dn_stations:
            if (station_id, 1) not in line.platform_names:
                platform_name = line.platform_names[(2 * len(line.up_stations) - station_id - 1, 0)]
                line.platform_names[(station_id, 1)] = platform_name

    # generate feasible routes (short turning routes needs to be determined by passenger flow)
    for key, line in lines.items():
        # up direction full route
        line.routes.append(line.up_stations)
        # down direction full route
        line.routes.append(line.dn_stations)

    return lines


def read_transect_flows(folder_path, type_name, lines):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    sectional_flows = {line_id: {} for line_id in lines.keys()}  # section flow of all lines, key: line_id
    # for each line
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        _, _, line_id, _, direction_flag = csv_file.split(".")[0].split("-")
        line = lines[line_id]
        sectional_flow = []  # section flow for this line and this direction
        with open(csv_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                row_data = {}
                for i in range(1, len(row)):
                    sta_id = i - 1 if direction_flag == '0' else i - 1 + len(line.up_stations)
                    row_data[(sta_id, sta_id + 1)] = int(row[i])
                sectional_flow.append(row_data)

        sectional_flows[line_id][direction_flag] = sectional_flow

    return PassengerFlow(type_name, sectional_flows)


def set_line_short_routes(lines, passenger_flows):
    for line_id, line in lines.items():
        # check if it has middle stations that connect with depots
        mid_station_conn_depots = [i for i in line.stations_conn_depots
                                   if i not in [line.up_stations[0], line.up_stations[-1],
                                                line.dn_stations[0], line.dn_stations[-1]]]
        if len(mid_station_conn_depots) == 0:
            continue
        flow_data = passenger_flows.sectional_flows[line.line_id]
        is_flow_unbalance = False  # to see if the flow during peak hours is fluctuating
        new_short_routes = []  # short routes need to be added if is_flow_unbalance is True

        for window in PEAK_HOURS:
            if is_flow_unbalance: break

            start_time_id, end_time_id = [int((t - START_TIME) / INTERVAL) for t in window]
            # up direction and down direction flows
            for key, rows in flow_data.items():
                # for each row (in flow_data) during peak hours
                if key == '0':  # up direction
                    mid_station_id = min(mid_station_conn_depots)
                    routes = [[i for i in range(line.up_stations[0], mid_station_id + 1)],
                              [i for i in range(mid_station_id, line.up_stations[-1] + 1)]]
                    avg_passenger_flows = [[], []]  # flows for two routes over several hours of peak hours
                    for t in range(start_time_id, end_time_id):
                        row = rows[t]
                        for k in range(0, len(routes)):
                            route = routes[k]
                            # average flow over the route
                            avg_passenger_flow = sum([int(row[(i, i + 1)]) for i in route[:-1]]) / (len(route) - 1)
                            avg_passenger_flows[k].append(avg_passenger_flow)
                    # average flow over peak hours and routes
                    avg_passenger_flows = [statistics.mean(avg_passenger_flows[0]),
                                           statistics.mean(avg_passenger_flows[-1])]
                    if max(avg_passenger_flows) > FLOW_FACTOR * min(avg_passenger_flows):
                        is_flow_unbalance = True
                        # generate short routes
                        short_route_id = avg_passenger_flows.index(max(avg_passenger_flows))
                        short_route = routes[short_route_id]
                        short_route_reverse = [2 * len(line.up_stations) - 1 - i for i in short_route][::-1]
                        new_short_routes += [short_route, short_route_reverse]
                        break  # breaks 'for key, rows in flow_data.items()'

                else:  # down direction
                    mid_station_id = max(mid_station_conn_depots)
                    routes = [[i for i in range(line.dn_stations[0], mid_station_id + 1)],
                              [i for i in range(mid_station_id, line.dn_stations[-1] + 1)]]
                    avg_passenger_flows = [[], []]  # flows for two routes over several hours of peak hours
                    for t in range(start_time_id, end_time_id):
                        row = rows[t]
                        for k in range(0, len(routes)):
                            route = routes[k]
                            # average flow over the route
                            avg_passenger_flow = sum(
                                [int(row[(i, i + 1)]) for i in route[:-1]]) / (len(route) - 1)
                            avg_passenger_flows[k].append(avg_passenger_flow)
                    # average flow over peak hours and routes
                    avg_passenger_flows = [statistics.mean(avg_passenger_flows[0]),
                                           statistics.mean(avg_passenger_flows[-1])]
                    if max(avg_passenger_flows) > FLOW_FACTOR * min(avg_passenger_flows):
                        is_flow_unbalance = True
                        # generate short routes
                        short_route_id = avg_passenger_flows.index(max(avg_passenger_flows))
                        short_route = routes[short_route_id]
                        short_route_reverse = [2 * len(line.up_stations) - 1 - i for i in short_route][::-1]
                        new_short_routes += [short_route, short_route_reverse]
                        break  # breaks 'for key, rows in flow_data.items()'

        if is_flow_unbalance:
            # generate short routes
            line.routes += new_short_routes


def read_arrival_rates(folder_path, lines, passenger_flows):
    for filename in os.listdir(folder_path):
        _, _, line_id = filename.split(".")[0].split("-")
        line = lines[line_id]
        file_path = os.path.join(folder_path, filename)
        # read all sheets
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        # read every sheet
        for sheet_name, df in all_sheets.items():
            column_num = df.shape[1]
            # for every row
            for index, row in df.iterrows():
                station_id = int(row[0]) if sheet_name == 'up' else 2 * len(line.up_stations) - 1 - int(row[0])
                for j in range(1, column_num):
                    start_t = (j - 1) * INTERVAL
                    end_t = start_t + INTERVAL
                    # calculate arrival rate (for each second)
                    arrival_rate = row[j] / INTERVAL
                    passenger_flows.d.update(
                        {(int(line_id), station_id, t): arrival_rate for t in range(start_t, end_t)})


def read_alight_rates(folder_path, lines, passenger_flows):
    for filename in os.listdir(folder_path):
        _, _, line_id = filename.split(".")[0].split("-")
        line = lines[line_id]
        sectional_flows = passenger_flows.sectional_flows[line_id]
        file_path = os.path.join(folder_path, filename)
        # read all sheets
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        # read every sheet
        for sheet_name, df in all_sheets.items():
            if 'alight' not in sheet_name:
                continue
            column_num = df.shape[1]
            # transect volume in this direction
            transect_volume = sectional_flows['0'] if sheet_name == 'alight-up' \
                else sectional_flows['1']
            # for every row
            for index, row in df.iterrows():
                station_id = int(row[0]) if sheet_name == 'alight-up' else 2 * len(line.up_stations) - 1 - int(row[0])
                for j in range(1, column_num):
                    start_t = (j - 1) * INTERVAL
                    end_t = start_t + INTERVAL
                    if station_id in [line.up_stations[0], line.dn_stations[0]]:
                        alight_rate = 0  # the first station no one alights the train
                    elif station_id in [line.up_stations[-1], line.dn_stations[-1]]:
                        alight_rate = 1  # the last station all passengers alight the train
                    else:
                        front_transect_volume = transect_volume[j - 1][(station_id - 1, station_id)]
                        alight_num = min(front_transect_volume, int(row[j]))
                        alight_rate = alight_num / max(front_transect_volume, 0.000001)
                    passenger_flows.theta.update(
                        {(int(line_id), station_id, t): alight_rate for t in range(start_t, end_t)})


def read_transfer_rates(folder_path, lines, passenger_flows):
    for filename in os.listdir(folder_path):
        station_name = filename.split('.')[0]
        file_path = os.path.join(folder_path, filename)
        # read sheet
        df = pd.read_excel(file_path)
        column_num = df.shape[1]
        for index, row in df.iterrows():
            line_id1, dir_flag1, line_id2, dir_flag2 = row[0].split(',')
            line1 = lines[line_id1]
            line2 = lines[line_id2]
            station_id1 = [sta_id for (sta_id, direction), value in line1.platform_names.items()
                           if direction == int(dir_flag1) and value == station_name][0]
            station_id2 = [sta_id for (sta_id, direction), value in line2.platform_names.items()
                           if direction == int(dir_flag2) and value == station_name][0]
            # for each column
            for j in range(1, column_num):
                start_t = (j - 1) * INTERVAL
                end_t = start_t + INTERVAL
                transfer_rate = float(row[j])
                passenger_flows.phi.update(
                    {(int(line_id1), station_id1, int(line_id2), station_id2, t): transfer_rate for t in
                     range(start_t, end_t)})
