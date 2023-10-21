import csv
import math
import os
import random
import statistics
import time
from collections import OrderedDict
import shutil

import pandas as pd

from Depot import *
from PassengerFlow import *
from RailLine import *
from Timetable import *
from TrainService import *
import copy
from sortedcontainers import SortedDict

INTERVAL = 30  # minutes
START_TIME = 5 * 60  # minutes
END_TIME = 24 * 60
PEAK_HOURS = [(7 * 60, 9 * 60), (17 * 60, 19 * 60)]
FLOW_FACTOR = 1.5
HEADWAY_MIN = 180  # seconds
HEADWAY_MAX = 600
HEADWAY_POOL = [i for i in range(HEADWAY_MIN, HEADWAY_MAX + 1, int((HEADWAY_MAX - HEADWAY_MIN) / 2))]
TRAIN_CAPACITY = 1200  # loading capacity
TRANSFER_SECS = 120  # fixed transfer time
TURNBACK_MAX_SECS = 10 * 60  # maximum turnback time
TURNBACK_MIN_SECS = 2 * 60  # minimum turnback time
LOOPLINE_CONN_MIN_SECS = 0 * 60  # minimum connection time for loop line
LOOPLINE_CONN_MAX_SECS = 5 * 60  # maximum connection time for loop line
SHORT_ROUTE_WEIGHT = 0.2
FULL_ROUTE_WEIGHT = 1 - SHORT_ROUTE_WEIGHT

# model parameters
TIME_PERIOD = 60  # 1 hour
N = [i for i in range(START_TIME, END_TIME + 1, TIME_PERIOD)]


def read_lines(folder_path):
    csv_files = [
        file
        for file in os.listdir(folder_path)
        if file.endswith(".csv") and file != "general_info.csv"
    ]

    if not csv_files:
        print("No CSV files found in the given folder.")
        return

    lines = {}
    # read general_info.csv
    csv_file_path = os.path.join(folder_path, "general_info.csv")
    with open(csv_file_path, "r", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            rail_line = RailLine(row)
            lines[rail_line.line_id] = rail_line

    # read other files
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        line_id = csv_file.split(".")[0].split("-")[2]
        with open(csv_file_path, "r", newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            rail_line = lines[line_id]
            for row in reader:
                (
                    station_name,
                    station_id,
                    dwell_time,
                    runtime,
                    is_transfer_station,
                    conn_depot_id,
                ) = row

                rail_line.dwells[int(station_id)] = int(dwell_time)
                if int(station_id) not in rail_line.up_stations:
                    rail_line.up_stations.append(int(station_id))

                    if conn_depot_id != "":
                        rail_line.conn_depots.append(int(conn_depot_id))
                        rail_line.stations_conn_depots[int(station_id)] = int(
                            conn_depot_id
                        )

                    # add sections
                    if runtime != "":
                        rail_line.runtimes[
                            (int(station_id), int(station_id) + 1)
                        ] = int(runtime)
                else:
                    station_cnt = len(rail_line.up_stations)
                    temp_runtime = rail_line.runtimes[(station_cnt - 1, station_cnt)]
                    del rail_line.runtimes[(station_cnt - 1, station_cnt)]
                    rail_line.runtimes[(station_cnt - 1, 0)] = temp_runtime
                    rail_line.is_loop_line = True

                # add up direction platform names
                if (station_id, 0) not in rail_line.platform_names:
                    rail_line.platform_names[(int(station_id), 0)] = station_name

    # generate values for down direction
    for key, line in lines.items():
        # generate stations
        line.dn_stations = [
            i for i in range(len(line.up_stations), 2 * len(line.up_stations))
        ]
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
                platform_name = line.platform_names[
                    (2 * len(line.up_stations) - station_id - 1, 0)
                ]
                line.platform_names[(station_id, 1)] = platform_name
        # set down direction dwells
        line.dwells.update(
            {
                2 * len(line.up_stations) - sta_id - 1: value
                for sta_id, value in line.dwells.items()
            }
        )

    # generate feasible routes (short turning routes needs to be determined by passenger flow)
    for key, line in lines.items():
        if not line.is_loop_line:
            # up direction full route
            line.routes.append(line.up_stations)
            line.up_routes.append((line.up_stations, line.get_runtime(line.up_stations[0], line.up_stations[-1])))
            # down direction full route
            line.routes.append(line.dn_stations)
            line.dn_routes.append((line.dn_stations, line.get_runtime(line.dn_stations[0], line.dn_stations[-1])))
        else:
            # up direction full route
            if line.up_stations[0] in line.stations_conn_depots:  # first station connect with depot
                temp_route = line.up_stations + [line.up_stations[0]]
                temp_runtime = line.get_runtime(temp_route[0], temp_route[-2]) + \
                               line.runtimes[(temp_route[-2], temp_route[0])]
                line.routes.append(temp_route)
                line.up_routes.append((temp_route, temp_runtime))
            elif line.up_stations[-1] in line.stations_conn_depots:  # last station connect with depot
                temp_route = [line.up_stations[-1]] + line.up_stations
                temp_runtime = line.get_runtime(temp_route[1], temp_route[-1]) + \
                               line.runtimes[(temp_route[0], temp_route[1])]
                line.routes.append(temp_route)
                line.up_routes.append((temp_route, temp_runtime))

            # dn direction full route
            if line.up_stations[0] in line.stations_conn_depots:  # first station connect with depot
                temp_route = [line.dn_stations[-1]] + line.dn_stations
                temp_runtime = line.get_runtime(temp_route[1], temp_route[-1]) + \
                               line.runtimes[(temp_route[0], temp_route[1])]
                line.routes.append(temp_route)
                line.dn_routes.append((temp_route, temp_runtime))
            elif line.up_stations[-1] in line.stations_conn_depots:  # last station connect with depot
                temp_route = line.dn_stations + [line.dn_stations[0]]
                temp_runtime = line.get_runtime(temp_route[0], temp_route[-2]) + \
                               line.runtimes[(temp_route[-2], temp_route[-1])]
                line.routes.append(temp_route)
                line.dn_routes.append((temp_route, temp_runtime))
    return lines


def read_depots(folder_path, lines):
    csv_files = [
        file
        for file in os.listdir(folder_path)
        if file.endswith(".csv") and file != "general_info.csv"
    ]

    if not csv_files:
        print("No CSV files found in the given folder.")
        return

    depots = {}
    # read files
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        depot_id = csv_file.split(".")[0].split("-")[2]
        depot = Depot(depot_id)
        with open(csv_file_path, "r", newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                _, capacity, conn_line_id, conn_station_id, runtime = row
                if capacity != "":
                    depot.capacity = int(capacity)
                depot.conn_lines.append(int(conn_line_id))
                depot.conn_stations[int(conn_line_id)].append(int(conn_station_id))
                reverse_station_id = 2 * len(lines[conn_line_id].up_stations) - int(conn_station_id) - 1
                depot.conn_stations[int(conn_line_id)].append(reverse_station_id)
                depot.runtime[(int(conn_line_id), int(conn_station_id))] = int(
                    runtime
                )
        depots[depot_id] = depot

    return depots


def read_transect_flows(folder_path, type_name, lines):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
    sectional_flows = {
        line_id: {} for line_id in lines.keys()
    }  # section flow of all lines, key: line_id
    # for each line
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        _, _, line_id, _, direction_flag = csv_file.split(".")[0].split("-")
        line = lines[line_id]
        sectional_flow = []  # section flow for this line and this direction
        with open(csv_file_path, "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                row_data = {}
                for i in range(1, len(row)):
                    sta_id = (
                        i - 1
                        if direction_flag == "0"
                        else i - 1 + len(line.up_stations)
                    )
                    row_data[(sta_id, sta_id + 1)] = int(row[i])
                sectional_flow.append(row_data)

        sectional_flows[line_id][direction_flag] = sectional_flow

    return PassengerFlow(type_name, sectional_flows)


def set_line_short_routes(lines, passenger_flows):
    for line_id, line in lines.items():
        # check if it has middle stations that connect with depots
        mid_station_conn_depots = [
            i
            for i in line.stations_conn_depots
            if i
               not in [
                   line.up_stations[0],
                   line.up_stations[-1],
                   line.dn_stations[0],
                   line.dn_stations[-1],
               ]
        ]
        if len(mid_station_conn_depots) == 0:
            continue
        flow_data = passenger_flows.sectional_flows[line.line_id]
        is_flow_unbalance = False  # to see if the flow during peak hours is fluctuating
        new_short_routes = ([])  # short routes need to be added if is_flow_unbalance is True

        for window in PEAK_HOURS:
            if is_flow_unbalance:
                break
            start_time_id, end_time_id = [
                int((t - START_TIME) / INTERVAL) for t in window]
            # up direction and down direction flows
            for key, rows in flow_data.items():
                # for each row (in flow_data) during peak hours
                if key == "0":  # up direction
                    mid_station_id = min(mid_station_conn_depots)
                    routes = [
                        [i for i in range(line.up_stations[0], mid_station_id + 1)],
                        [i for i in range(mid_station_id, line.up_stations[-1] + 1)],
                    ]
                    avg_passenger_flows = [
                        [],
                        [],
                    ]  # flows for two routes over several hours of peak hours
                    for t in range(start_time_id, end_time_id):
                        row = rows[t]
                        for k in range(0, len(routes)):
                            route = routes[k]
                            # average flow over the route
                            avg_passenger_flow = sum(
                                [int(row[(i, i + 1)]) for i in route[:-1]]
                            ) / (len(route) - 1)
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
                    routes = [
                        [i for i in range(line.dn_stations[0], mid_station_id + 1)],
                        [i for i in range(mid_station_id, line.dn_stations[-1] + 1)],
                    ]
                    avg_passenger_flows = [
                        [],
                        [],
                    ]  # flows for two routes over several hours of peak hours
                    for t in range(start_time_id, end_time_id):
                        row = rows[t]
                        for k in range(0, len(routes)):
                            route = routes[k]
                            # average flow over the route
                            avg_passenger_flow = sum(
                                [int(row[(i, i + 1)]) for i in route[:-1]]
                            ) / (len(route) - 1)
                            avg_passenger_flows[k].append(avg_passenger_flow)
                    # average flow over peak hours and routes
                    avg_passenger_flows = [
                        statistics.mean(avg_passenger_flows[0]),
                        statistics.mean(avg_passenger_flows[-1]),
                    ]
                    if max(avg_passenger_flows) > FLOW_FACTOR * min(
                            avg_passenger_flows
                    ):
                        is_flow_unbalance = True
                        # generate short routes
                        short_route_id = avg_passenger_flows.index(
                            max(avg_passenger_flows)
                        )
                        short_route = routes[short_route_id]
                        short_route_reverse = [
                                                  2 * len(line.up_stations) - 1 - i for i in short_route
                                              ][::-1]
                        new_short_routes += [short_route, short_route_reverse]
                        break  # breaks 'for key, rows in flow_data.items()'

        # for normal line
        if not line.is_loop_line and is_flow_unbalance:
            # generate short routes
            line.routes += new_short_routes
            for short_route in new_short_routes:
                route_runtime = line.get_runtime(short_route[0], short_route[-1])
                if short_route[0] <= line.up_stations[-1]:
                    line.up_routes.append((short_route, route_runtime))
                else:
                    line.dn_routes.append((short_route, route_runtime))
        # for loop line
        elif line.is_loop_line and len(line.stations_conn_depots) == 4:
            # generate short routes
            station_list = sorted(list(line.stations_conn_depots.keys()))
            new_short_routes = [[i for i in range(station_list[0], station_list[1] + 1)],
                                [i for i in range(station_list[2], station_list[3] + 1)]]
            line.routes += new_short_routes
            for short_route in new_short_routes:
                route_runtime = line.get_runtime(short_route[0], short_route[-1])
                if short_route[0] <= line.up_stations[-1]:
                    line.up_routes.append((short_route, route_runtime))
                else:
                    line.dn_routes.append((short_route, route_runtime))


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
                station_id = (
                    int(row[0])
                    if sheet_name == "up"
                    else 2 * len(line.up_stations) - 1 - int(row[0])
                )
                for j in range(1, column_num):
                    start_t = (j - 1) * INTERVAL
                    end_t = start_t + INTERVAL
                    # calculate arrival rate (for each second)
                    arrival_rate = row[j] / INTERVAL
                    passenger_flows.d.update(
                        {
                            (int(line_id), station_id, t): arrival_rate
                            for t in range(start_t, end_t)
                        }
                    )


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
            if "alight" not in sheet_name:
                continue
            column_num = df.shape[1]
            # transect volume in this direction
            transect_volume = (
                sectional_flows["0"]
                if sheet_name == "alight-up"
                else sectional_flows["1"]
            )
            # for every row
            for index, row in df.iterrows():
                station_id = (
                    int(row[0])
                    if sheet_name == "alight-up"
                    else 2 * len(line.up_stations) - 1 - int(row[0])
                )
                for j in range(1, column_num):
                    start_t = (j - 1) * INTERVAL
                    end_t = start_t + INTERVAL
                    if station_id in [line.up_stations[0], line.dn_stations[0]]:
                        alight_rate = 0  # the first station no one alights the train
                    elif station_id in [line.up_stations[-1], line.dn_stations[-1]]:
                        alight_rate = (
                            1  # the last station all passengers alight the train
                        )
                    else:
                        front_transect_volume = transect_volume[j - 1][
                            (station_id - 1, station_id)
                        ]
                        alight_num = min(front_transect_volume, int(row[j]))
                        alight_rate = alight_num / max(front_transect_volume, 0.000001)
                    passenger_flows.theta.update(
                        {
                            (int(line_id), station_id, t): alight_rate
                            for t in range(start_t, end_t)
                        }
                    )


def read_transfer_rates(folder_path, lines, passenger_flows):
    for filename in os.listdir(folder_path):
        station_name = filename.split(".")[0]
        file_path = os.path.join(folder_path, filename)
        # read sheet
        df = pd.read_excel(file_path)
        column_num = df.shape[1]
        for index, row in df.iterrows():
            line_id1, dir_flag1, line_id2, dir_flag2 = row[0].split(",")
            line1 = lines[line_id1]
            line2 = lines[line_id2]
            station_id1 = [
                sta_id
                for (sta_id, direction), value in line1.platform_names.items()
                if direction == int(dir_flag1) and value == station_name
            ][0]
            station_id2 = [
                sta_id
                for (sta_id, direction), value in line2.platform_names.items()
                if direction == int(dir_flag2) and value == station_name
            ][0]
            # for each column
            for j in range(1, column_num):
                start_t = (j - 1) * INTERVAL
                end_t = start_t + INTERVAL
                transfer_rate = float(row[j])
                passenger_flows.phi.update(
                    {
                        (
                            int(line_id1),
                            station_id1,
                            int(line_id2),
                            station_id2,
                            t,
                        ): transfer_rate
                        for t in range(start_t, end_t)
                    }
                )
                # save transfer connections
                temp_key = (int(line_id1), station_id1, int(line_id2), station_id2)
                if temp_key not in line1.transfer_pairs:
                    line1.transfer_pairs.append(temp_key)


def gen_timetables(iter_max, lines, depots, passenger_flows):
    root_folder = "./test/"
    if os.path.exists(root_folder):
        shutil.rmtree(root_folder)
        os.mkdir(root_folder)
    else:
        os.mkdir(root_folder)

    random.seed(2023)

    timetable_pool = []
    weights_dict = {
        (l, n, d): [100 for i in HEADWAY_POOL] for l in lines.keys() for n in N for d in [0, 1]
    }  # key l,n,d
    for i in range(0, iter_max):
        print("iter: " + str(i))
        # generate a timetable for the network
        timetable_net = {l: Timetable(l) for l in lines.keys()}  # key: line_id

        arr_slots = {}  # key:(l,sta_id), the last arrive time of each station
        for l, line in lines.items():
            for sta_id in line.up_stations + line.dn_stations:
                arr_slots[l, sta_id] = START_TIME * 60

        service_nums = {l: 0 for l in lines.keys()}
        last_routes = {(line_id, dir_id): (None, None) for line_id in lines.keys()
                       for dir_id in [0, 1]}  # key: (line_id,direction) value: (sta_start,sta_end)

        # depots_dep_times = {i: SortedDict() for i in depots.keys()}
        # depots_arr_times = {i: SortedDict() for i in depots.keys()}

        # for each time window
        for n in N:
            is_in_peak_hour = (
                True if (PEAK_HOURS[0][0] <= n < PEAK_HOURS[0][1]) or (PEAK_HOURS[1][0] <= n < PEAK_HOURS[1][1])
                else False)
            time_span = [k for k in range(n, n + TIME_PERIOD)]

            for l, line in lines.items():  # for each line
                timetable = timetable_net[l]
                for route in line.routes:
                    for sta_id in [route[0], route[-1]]:
                        if not line.is_loop_line:
                            physic_sta_id = min(sta_id, 2 * len(line.up_stations) - 1 - sta_id)
                            timetable.services_queues.setdefault(physic_sta_id, {"from": [], "to": []})
                        else:
                            timetable.services_queues.setdefault(sta_id, {"from": [], "to": []})
                start_time_secs = time_span[0] * 60
                headway_up = roulette_selection(HEADWAY_POOL, weights_dict[line.line_id, n, 0])
                headway_dn = roulette_selection(HEADWAY_POOL, weights_dict[line.line_id, n, 1])
                # add upstream and downstream services
                while True:
                    added_servs = []
                    # upstream *******************
                    up_route_pool, dedicated_turn_back_servs, desired_arr_times = (
                        process_routes(line, line.up_routes, arr_slots,
                                       timetable, headway_up, time_span))
                    # select route: if full-length and short-length routes are all available
                    # (1) peak hour: alternatively select these routes
                    # (2) off-peak hour: only select full-length route
                    # if only one route: select this route
                    # if no available routes: skip

                    # select route from upstream available routes
                    route = up_route_pool[0] if len(up_route_pool) == 1 else None
                    if len(up_route_pool) > 0:
                        if not line.is_loop_line:
                            if len(up_route_pool) > 1:
                                # alternatively select
                                if is_in_peak_hour:
                                    route = up_route_pool[0] if \
                                        (last_routes[line.line_id, 0] != (up_route_pool[0][0], up_route_pool[0][-1])) \
                                        else up_route_pool[-1]
                                # only select full-length route
                                else:
                                    route = up_route_pool[0] if \
                                        (up_route_pool[0][0], up_route_pool[0][-1]) == (
                                            line.up_stations[0], line.up_stations[-1]) else up_route_pool[-1]
                        # loop line
                        else:
                            if len(up_route_pool) > 1:
                                weights = [SHORT_ROUTE_WEIGHT, FULL_ROUTE_WEIGHT] if len(up_route_pool[0]) < len(
                                    up_route_pool[-1]) else [FULL_ROUTE_WEIGHT,
                                                             SHORT_ROUTE_WEIGHT]
                                route = roulette_selection(up_route_pool, weights)

                    if route is not None:
                        first_arr_time = max(start_time_secs, arr_slots[l, route[0]] + headway_up)
                        if (route[0], route[-1]) in desired_arr_times.keys():
                            first_arr_time = desired_arr_times[(route[0], route[-1])]
                        service, valid = construct_service(route, service_nums, 0, line, first_arr_time, time_span,
                                                           timetable, arr_slots)
                        if valid:
                            # save the info of connecting depots for this service
                            for term_sta in [route[0], route[-1]]:
                                if term_sta in line.stations_conn_depots.keys():
                                    if term_sta == route[0]:
                                        service.from_depot_id = line.stations_conn_depots[term_sta]
                                    else:
                                        service.to_depot_id = line.stations_conn_depots[term_sta]

                            added_servs.append(service)
                            last_routes[line.line_id, 0] = (route[0], route[-1])
                            # update turn-back connections
                            if (route[0], route[-1]) in dedicated_turn_back_servs.keys():
                                direction, serv_id = dedicated_turn_back_servs[(route[0], route[-1])]
                                if direction == 'from':
                                    service.front_service = serv_id
                                    timetable.services[serv_id].next_service = service.id
                                else:
                                    service.next_service = serv_id
                                    timetable.services[serv_id].front_service = service.id
                            # update timetable.services_queues
                            if service.front_service == -1:
                                if not line.is_loop_line:
                                    physic_sta_id = min(route[0], 2 * len(line.up_stations) - 1 - route[0])
                                    timetable.services_queues[physic_sta_id]['to'].append(service.id)
                                else:
                                    timetable.services_queues[route[0]]['to'].append(service.id)
                            if service.next_service == -1:
                                if not line.is_loop_line:
                                    physic_sta_id = min(route[-1], 2 * len(line.up_stations) - 1 - route[-1])
                                    timetable.services_queues[physic_sta_id]['from'].append(service.id)
                                else:
                                    timetable.services_queues[route[-1]]['from'].append(service.id)

                    # downstream ********************
                    dn_route_pool, dedicated_turn_back_servs, desired_arr_times = (
                        process_routes(line, line.dn_routes, arr_slots,
                                       timetable, headway_dn, time_span))

                    # select route from downstream available routes
                    route = dn_route_pool[0] if len(dn_route_pool) == 1 else None
                    if len(dn_route_pool) > 0:
                        if not line.is_loop_line:
                            if len(dn_route_pool) > 1:
                                # alternatively select
                                if is_in_peak_hour:
                                    route = dn_route_pool[0] if \
                                        (last_routes[line.line_id, 1] != (
                                            dn_route_pool[0][0], dn_route_pool[0][-1])) \
                                        else dn_route_pool[-1]
                                # only select full-length route
                                else:
                                    route = dn_route_pool[0] if \
                                        (dn_route_pool[0][0], dn_route_pool[0][-1]) == (
                                            line.dn_stations[0], line.dn_stations[-1]) else dn_route_pool[-1]
                        # loop line
                        else:
                            if len(dn_route_pool) > 1:
                                weights = [SHORT_ROUTE_WEIGHT, FULL_ROUTE_WEIGHT] if len(dn_route_pool[0]) < len(
                                    dn_route_pool[-1]) else [FULL_ROUTE_WEIGHT,
                                                             SHORT_ROUTE_WEIGHT]
                                route = roulette_selection(dn_route_pool, weights)

                    if route is not None:
                        first_arr_time = max(start_time_secs, arr_slots[l, route[0]] + headway_dn)
                        if (route[0], route[-1]) in desired_arr_times.keys():
                            first_arr_time = desired_arr_times[(route[0], route[-1])]
                        service, valid = construct_service(route, service_nums, 1, line, first_arr_time, time_span,
                                                           timetable, arr_slots)

                        if valid:
                            # save the info of connecting depots for this service
                            for term_sta in [route[0], route[-1]]:
                                if term_sta in line.stations_conn_depots.keys():
                                    if term_sta == route[0]:
                                        service.from_depot_id = line.stations_conn_depots[term_sta]
                                    else:
                                        service.to_depot_id = line.stations_conn_depots[term_sta]

                            added_servs.append(service)
                            last_routes[line.line_id, 1] = (route[0], route[-1])
                            # update turn-back connections
                            if (route[0], route[-1]) in dedicated_turn_back_servs.keys():
                                direction, serv_id = dedicated_turn_back_servs[(route[0], route[-1])]
                                if direction == 'from':
                                    service.front_service = serv_id
                                    timetable.services[serv_id].next_service = service.id
                                else:
                                    service.next_service = serv_id
                                    timetable.services[serv_id].front_service = service.id
                            # update timetable.services_queues
                            if service.front_service == -1:
                                if not line.is_loop_line:
                                    physic_sta_id = min(route[0], 2 * len(line.up_stations) - 1 - route[0])
                                    timetable.services_queues[physic_sta_id]['to'].append(service.id)
                                else:
                                    timetable.services_queues[route[0]]['to'].append(service.id)
                            if service.next_service == -1:
                                if not line.is_loop_line:
                                    physic_sta_id = min(route[-1], 2 * len(line.up_stations) - 1 - route[-1])
                                    timetable.services_queues[physic_sta_id]['from'].append(service.id)
                                else:
                                    timetable.services_queues[route[-1]]['from'].append(service.id)

                    if len(added_servs) == 0:
                        break  # breaks the While loop

        # simulate passenger flows and calculate service quality
        # start_time = time.time()
        # passenger_flow_simulate(timetable_net, lines, passenger_flows)
        # avg_serv_quality = calculate_service_quality(timetable_net, lines)
        avg_serv_quality = 0.5
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"程序执行时间: {execution_time:.4f} 秒")

        delete_services(timetable_net, lines, depots)

        # generate vehicle circulations
        gen_vehicle_circulation(timetable_net, lines)

        # # generate rolling stock allocation plan
        # gen_rs_allocation_plan(depots, depots_dep_times, depots_dep_times)

        for depot in depots.values():
            if depot.maximum_flow > depot.capacity:
                ssda = 0

        timetable_pool.append((timetable_net, avg_serv_quality))

        export_timetable(root_folder, timetable_net, lines)

    return timetable_pool


def roulette_selection(pool, weights=None):
    if not weights:
        weights = [1 / len(pool) for _ in range(len(pool))]
    else:
        total_weight = sum(weights)
        weights = [weight / total_weight for weight in weights]

    r = random.random()
    cumulative_weight = 0

    for i, weight in enumerate(weights):
        cumulative_weight += weight
        if r <= cumulative_weight:
            return pool[i]

    return pool[-1]


def export_timetable(folder, timetable_net, lines):
    for l, timetable in timetable_net.items():
        line = lines[l]
        folder_name = folder + str(l) + "/"

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        # save each service
        for serv_id, service in timetable.services.items():
            csv_file_path = folder_name + "service[{}].csv".format(str(serv_id))
            csv_data = [["line_idx", "sta_idx", "dir", "type", "time", "depot_idx"]]
            for i in range(0, len(service.route)):
                if service.direction == 0 and line.is_loop_line and len(service.route) > len(line.up_stations):
                    if i == 0:
                        continue
                if service.direction == 1 and line.is_loop_line and len(service.route) > len(line.up_stations):
                    if i == len(service.route) - 1:
                        continue
                sta_id = service.route[i]
                arr_time = service.arrs[i]
                dep_time = service.deps[i]
                direction = service.direction
                if direction == 1:
                    sta_id = 2 * len(line.up_stations) - sta_id - 1
                csv_data.append([l, sta_id, direction, "arr", arr_time, -1])
                csv_data.append([l, sta_id, direction, "dep", dep_time, -1])

            csv_data.insert(1, [-1, -1, -1, "depot_out", service.arrs[0] - 120, 0])
            csv_data.append([-1, -1, -1, "depot_in", service.deps[-1] + 120, 0])

            with open(csv_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(csv_data)
        # generate a file to save turn-back connections
        csv_file_path = folder_name + "connections.csv"
        csv_data = [["serv_sequence"]]
        for first_serv_id, serv_sequence in timetable.turn_back_connections.items():
            result_string = '-'.join(map(str, serv_sequence))
            csv_data.append([result_string])
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)


def passenger_flow_simulate(timetable_net, lines, passenger_flows):
    arrive_slots = defaultdict(list)
    depart_slots = defaultdict(list)
    core_slots = defaultdict(list)

    # !!! t (seconds) is starting from START_TIME
    transfer_volume = {}  # key (l2,s2)
    transfer_volume_points = {}  # key (l2,s2)

    # set core_slots, arrive_slots and depart_slots
    for l, timetable in timetable_net.items():
        for service_id, service in timetable.services.items():
            for t in service.arrs:
                if t == 86400:
                    ssda = 0
                arrive_slots[t].append((l, service_id))
                if "arr" not in core_slots[t]:
                    core_slots[t].append("arr")
            for t in service.deps:
                if t == 86400:
                    ssda = 0
                depart_slots[t].append((l, service_id))
                if "dep" not in core_slots[t]:
                    core_slots[t].append("dep")
    # sort
    arrive_slots = OrderedDict(sorted(arrive_slots.items()))
    depart_slots = OrderedDict(sorted(depart_slots.items()))
    core_slots = OrderedDict(sorted(core_slots.items()))

    # initialize p_wait, p_train, p_alight, p_transfer
    for l, timetable in timetable_net.items():
        timetable.p_wait = {
            sta_id: [(START_TIME * 60, 0, "init")]
            for sta_id in (lines[l].up_stations + lines[l].dn_stations)
        }
        timetable.p_train = {
            serv_id: {sta_id: 0 for sta_id in serv.route}
            for serv_id, serv in timetable.services.items()
        }
        # timetable.p_alight = {serv_id: {sta_id: 0 for sta_id in serv.route} for serv_id, serv in
        #                       timetable.services.items()}
        # timetable.p_transfer = {serv_id: {(s1, s2): 0 for (_, s1, _, s2) in lines[l].transfer_pairs}
        #                         for serv_id in timetable.services.keys()}

    # simulate passenger flows
    for t, action_list in core_slots.items():  # t is starting from START_TIME
        t_minute = math.floor((int(t) - START_TIME * 60) / 60)
        # this slot contains arrivals
        if "arr" in action_list:
            # calculate the alight passengers and transfer passengers
            for l, service_id in arrive_slots[t]:
                timetable = timetable_net[l]
                service = timetable.services[service_id]
                station_id = service.route[service.arrs.index(t)]
                alight_rate = passenger_flows.theta[int(l), station_id, t_minute]
                passenger_in_train = (
                    0
                    if station_id == service.route[0]
                    else timetable.p_train[service_id][service.route[service.arrs.index(t) - 1]]
                )
                alight_pas = math.floor(passenger_in_train * alight_rate)
                # update alight passengers
                # timetable.p_alight[service_id][station_id] = alight_pas
                # update transfer passengers
                for (l1, s1, l2, s2) in lines[l].transfer_pairs:
                    transfer_rate = passenger_flows.phi[l1, s1, l2, s2, t_minute]
                    # timetable.p_transfer[service_id][s1, l2, s2] = alight_pas * transfer_rate
                    # ignore 0
                    temp_value = math.floor(alight_pas * transfer_rate)
                    arrive_t = int(t) + TRANSFER_SECS
                    if temp_value > 0:
                        if (l2, s2) not in transfer_volume.keys():
                            transfer_volume[l2, s2] = []
                            transfer_volume_points[l2, s2] = 0
                        transfer_volume[l2, s2].append((arrive_t, temp_value))
                        # transfer_volume[l2, s2, arrive_t] = temp_value

            # calculate the waiting passengers
            for l, service_id in arrive_slots[t]:
                timetable = timetable_net[l]
                service = timetable.services[service_id]
                station_id = service.route[service.arrs.index(t)]
                # remaining passengers + new arriving passengers + transfer passengers
                pre_time_second, remaining_pas, _ = timetable.p_wait[station_id][-1]
                pre_time_minute = round((pre_time_second - START_TIME * 60) / 60)
                arriving_pas = math.ceil(
                    sum(
                        [
                            passenger_flows.d[int(l), station_id, t_slot]
                            for t_slot in range(pre_time_minute, t_minute)
                        ]
                    )
                )
                # calculate transfer volume
                transfer_pas = 0
                while True:
                    temp_key = (int(l), station_id)
                    if temp_key not in transfer_volume or not transfer_volume[temp_key]:
                        break
                    i = transfer_volume_points[temp_key]
                    if i > len(transfer_volume[temp_key]) - 1:
                        break
                    t_slot, value = transfer_volume[temp_key][i]
                    if t_slot > t:
                        break
                    elif pre_time_second < t_slot <= t:
                        transfer_pas += value
                        transfer_volume_points[temp_key] = i + 1

                wait_passengers = (
                        remaining_pas + arriving_pas + math.floor(transfer_pas)
                )
                timetable.p_wait[station_id].append((t, wait_passengers, "arr"))
                # timetable.sum_passenger_number += arriving_pas

        # this slot contains departures
        if "dep" in action_list:
            # calculate the waiting passengers and boarding passengers
            for l, service_id in depart_slots[t]:
                timetable = timetable_net[l]
                service = timetable.services[service_id]
                station_id = service.route[service.deps.index(t)]
                # remaining passengers + new arriving passengers + transfer passengers
                pre_time_second, remaining_pas, _ = timetable.p_wait[station_id][-1]
                pre_time_minute = round((pre_time_second - START_TIME * 60) / 60)
                arriving_pas = sum(
                    [
                        passenger_flows.d[int(l), station_id, t_slot]
                        for t_slot in range(pre_time_minute, t_minute)
                    ]
                )
                # calculate transfer volume
                transfer_pas = 0

                while True:
                    temp_key = (int(l), station_id)
                    if temp_key not in transfer_volume or not transfer_volume[temp_key]:
                        break
                    i = transfer_volume_points[temp_key]
                    if i > len(transfer_volume[temp_key]) - 1:
                        break
                    t_slot, value = transfer_volume[temp_key][i]
                    if t_slot > t:
                        break
                    elif pre_time_second < t_slot <= t:
                        transfer_pas += value
                        transfer_volume_points[temp_key] = i + 1

                wait_passengers = remaining_pas + arriving_pas + math.floor(transfer_pas)
                passenger_in_train = (
                    0
                    if station_id == service.route[0]
                    else timetable.p_train[service_id][service.route[service.deps.index(t) - 1]]
                )
                boarding_passengers = min(
                    wait_passengers, TRAIN_CAPACITY - passenger_in_train
                )
                # update en-route passengers
                timetable.p_train[service_id][station_id] = (
                        passenger_in_train + boarding_passengers
                )
                # update remaining passengers
                wait_passengers -= boarding_passengers
                timetable.p_wait[station_id].append((t, wait_passengers, "dep"))
                # timetable.sum_passenger_number += arriving_pas


def calculate_service_quality(timetable_net, lines):
    service_quality = 0
    for l, timetable in timetable_net.items():
        line = lines[l]
        for sta_id, wait_list in timetable.p_wait.items():
            for i in range(len(wait_list) - 1):
                avg_time_diff = 0.5 * (wait_list[i + 1][0] - wait_list[i][0])
                avg_passenger_number = 0.5 * (wait_list[i + 1][1] + wait_list[i][1])
                flag = wait_list[i + 1][2]
                if avg_passenger_number > 0:
                    timetable.sum_wait_time += avg_passenger_number * avg_time_diff
                    if flag == "arr":
                        timetable.sum_wait_time_min += (
                                0.5 * avg_passenger_number * HEADWAY_MIN
                        )
                        timetable.sum_wait_time_max += (
                                0.5 * avg_passenger_number * HEADWAY_MAX
                        )
                    elif flag == "dep":
                        temp_wait_time = (
                                0.5 * avg_passenger_number * line.dwells[sta_id])
                        timetable.sum_wait_time_min += temp_wait_time
                        timetable.sum_wait_time_max += temp_wait_time
        temp_serv_quality = (timetable.sum_wait_time - timetable.sum_wait_time_min) / (
                timetable.sum_wait_time_max - timetable.sum_wait_time_min
        )
        service_quality += temp_serv_quality
    return service_quality / len(lines)


def gen_vehicle_circulation(timetable_net, lines):
    for l, timetable in timetable_net.items():
        line = lines[l]
        services_queues = timetable.services_queues
        # start linking services
        for turn_back_platform in services_queues.keys():
            service_queue = services_queues[turn_back_platform]
            j_pivot = 0  # lies around the last i's minimum turnback time index
            for i in range(len(service_queue["from"])):
                from_service = timetable.services[service_queue["from"][i]]
                j = j_pivot
                while True:
                    if j == len(service_queue["to"]) or from_service.next_service != -1:
                        break
                    # for normal line
                    if not line.is_loop_line:
                        from_time = from_service.deps[-1]
                        to_service = timetable.services[service_queue["to"][j]]
                        to_time = to_service.arrs[0]
                        # less than the minimum turnback time, move j to the next
                        if to_time - from_time < TURNBACK_MIN_SECS:
                            j += 1
                            j_pivot = j
                        # larger than the maximum turnback time
                        elif to_time - from_time > TURNBACK_MAX_SECS:
                            break
                        # valid turnback time
                        else:
                            # to_service is already connected by other services
                            if to_service.front_service != -1:
                                j += 1
                            # is valid to have a connection
                            else:
                                from_service.next_service = to_service.id
                                to_service.front_service = from_service.id

                                if from_service.first_service == -1:
                                    from_service.first_service = from_service.id

                                to_service.first_service = from_service.first_service
                                to_service.front_service = from_service.id
                                from_service.next_service = to_service.id
                                j_pivot = j + 1
                    # for loop line
                    else:
                        # check the connection time
                        from_time = from_service.arrs[-1]
                        to_service = timetable.services[service_queue["to"][j]]
                        to_time = to_service.arrs[0]

                        if line.line_id == '1' and from_service.id == 47 and to_service.id == 92:
                            sdsada = 0
                        # only consider services with the same direction
                        if from_service.direction != to_service.direction:  # move j to the next
                            j += 1
                            j_pivot = j
                            continue
                        # less than the minimum turnback time, move j to the next
                        if to_time - from_time < LOOPLINE_CONN_MIN_SECS:
                            j += 1
                            j_pivot = j
                        # larger than the maximum turnback time
                        elif to_time - from_time > LOOPLINE_CONN_MAX_SECS:
                            break
                        # valid turnback time
                        else:
                            # to_service is already connected by other services
                            if to_service.front_service != -1:
                                j += 1
                            # is valid to have a connection
                            else:
                                from_service.next_service = to_service.id
                                to_service.front_service = from_service.id

                                if from_service.first_service == -1:
                                    from_service.first_service = from_service.id

                                to_service.first_service = from_service.first_service
                                to_service.front_service = from_service.id
                                from_service.next_service = to_service.id
                                j_pivot = j + 1

        # generate service journeys
        for service_id in timetable.up_services + timetable.dn_services:
            service = timetable.services[service_id]
            if service.front_service == -1:
                timetable.turn_back_connections[service_id] = [service_id]
        for first_serv_id in timetable.turn_back_connections.keys():
            temp_service = timetable.services[first_serv_id]
            while temp_service.next_service != -1:
                if temp_service.id != first_serv_id:
                    timetable.turn_back_connections[first_serv_id].append(
                        temp_service.id
                    )
                temp_service = timetable.services[temp_service.next_service]
            last_serv_id = temp_service.id
            if temp_service.id != first_serv_id:
                timetable.turn_back_connections[first_serv_id].append(temp_service.id)

            for serv_id in timetable.turn_back_connections[first_serv_id]:
                service = timetable.services[serv_id]
                service.last_service = last_serv_id
                if service.first_service != -1:
                    service.first_service = first_serv_id

        # cancel invalid services that start or terminate at station without connections to depots
        to_delete = []
        to_replace = []  # new and delete
        for first_serv_id, service_queue in timetable.turn_back_connections.items():
            # forward check
            forward_check_pass = False
            while len(service_queue) > 0 and not forward_check_pass:
                first_service = timetable.services[service_queue[0]]
                first_service.first_service = first_service.id
                first_serv_first_station = first_service.route[0]
                # if this station does not connect to this depot, cancel this service
                if first_serv_first_station not in line.stations_conn_depots:
                    del timetable.services[service_queue[0]]
                    del service_queue[0]
                else:
                    forward_check_pass = True
            # backward check
            backward_check_pass = False
            last_service_altered = False
            while len(service_queue) > 0 and not backward_check_pass:
                last_service = timetable.services[service_queue[-1]]
                last_serv_last_station = last_service.route[-1]
                # if this station does not connect to this depot, cancel this service
                if last_serv_last_station not in line.stations_conn_depots:
                    del timetable.services[service_queue[-1]]
                    del service_queue[-1]
                    last_service_altered = True
                else:
                    backward_check_pass = True

            if len(service_queue) == 0:
                to_delete.append(first_serv_id)
            else:
                if last_service_altered:
                    for serv_id in service_queue:
                        service = timetable.services[serv_id]
                        service.last_service = service_queue[-1]
                        if serv_id == service_queue[-1]:
                            service.next_service = -1
                if service_queue[0] != first_serv_id:  # update the key
                    to_replace.append([service_queue[0], first_serv_id])

        for key in to_delete:
            del timetable.turn_back_connections[key]
        for key_new, key_delete in to_replace:
            timetable.turn_back_connections[key_new] = timetable.turn_back_connections[key_delete]
            del timetable.turn_back_connections[key_delete]

        # for loop lines, separate services from the route
        if line.is_loop_line:
            separate_services = []
            for serv_id, serv in timetable.services.items():
                if serv.route[0] == serv.route[-1]:
                    separate_services.append(serv)
            serv_num = len(timetable.services)
            for serv in separate_services:
                # turn-back at the start
                if abs(serv.route[0] - serv.route[1]) != 1:
                    serv_new_follow = TrainService(serv_num, serv.direction, serv.route[1:], line.line_id)
                    serv_num += 1
                    # alter arrs and deps of serv_new_follow
                    serv.route = serv.route[:1]
                    serv_new_follow.arrs = serv.arrs[1:]
                    serv_new_follow.deps = serv.deps[1:]
                    serv.arrs = serv.arrs[:1]
                    serv.deps = serv.deps[:1]

                    servs_sequence = timetable.turn_back_connections[serv.first_service]
                    # this service is the last of vehicle, update "last_service" for all involving services
                    if serv.id == servs_sequence[-1]:
                        for temp_serv_id in servs_sequence:
                            timetable.services[temp_serv_id].last_service = serv_new_follow.id
                        serv_new_follow.last_service = serv_new_follow.id
                    else:
                        # add turn-back connections
                        temp_serv_id = servs_sequence[servs_sequence.index(serv.id) + 1]
                        timetable.services[temp_serv_id].front_service = serv_new_follow.id
                        serv_new_follow.next_service = temp_serv_id
                    # add turn-back connections
                    serv.next_service = serv_new_follow.id
                    serv_new_follow.front_service = serv.id
                    # insert
                    servs_sequence.insert(servs_sequence.index(serv.id) + 1, serv_new_follow.id)
                    # add this new service to timetable
                    timetable.services[serv_new_follow.id] = serv_new_follow
                # turn-back at the end
                else:
                    serv_new_lead = TrainService(serv_num, serv.direction, serv.route[:-1], line.line_id)
                    serv_num += 1
                    # alter arrs and deps of serv_new_follow
                    serv.route = serv.route[-1:]
                    serv_new_lead.arrs = serv.arrs[:-1]
                    serv_new_lead.deps = serv.deps[:-1]
                    serv.arrs = serv.arrs[-1:]
                    serv.deps = serv.deps[-1:]

                    servs_sequence = timetable.turn_back_connections[serv.first_service]
                    # this service is the first of vehicle, update "first_service" for all involving services
                    if serv.id == servs_sequence[0]:
                        for temp_serv_id in servs_sequence:
                            timetable.services[temp_serv_id].first_service = serv_new_lead.id
                        serv_new_lead.first_service = serv_new_lead.id
                    else:
                        # add turn-back connections
                        temp_serv_id = servs_sequence[servs_sequence.index(serv.id) - 1]
                        timetable.services[temp_serv_id].next_service = serv_new_lead.id
                        serv_new_lead.front_service = temp_serv_id
                    # add turn-back connections
                    serv.front_service = serv_new_lead.id
                    serv_new_lead.next_service = serv.id
                    # insert
                    servs_sequence.insert(servs_sequence.index(serv.id), serv_new_lead.id)
                    # add this new service to timetable
                    timetable.services[serv_new_lead.id] = serv_new_lead
                    # update timetable.turn_back_connections, if new service is the first service
                    if serv_new_lead.first_service == serv_new_lead.id:
                        timetable.turn_back_connections[serv_new_lead.id] = timetable.turn_back_connections \
                            .pop(serv_new_lead.next_service)

            # update last service
            for fir_serv_id, service_queue in timetable.turn_back_connections.items():
                last_serv_id = service_queue[-1]
                for serv_id in service_queue[:-1]:
                    timetable.services[serv_id].last_service = last_serv_id


def gen_rs_allocation_plan(depots, depots_dep_times, depots_arr_times):
    for depot_id, depot in depots.items():
        depot_max_flow = calculate_depot_max_flow(depots_dep_times, depots_arr_times, depot_id)
        depot.maximum_flow = depot_max_flow


def process_routes(line, routes, arr_slots, timetable, headway, timespan):
    route_pool = []
    dedicated_turn_back_servs = {}  # key: (route[0],route[-1])   value: (direction, serv_id)
    desired_arr_times = {}  # key: (route[0],route[-1])   value: arrive_time

    end_time_secs = timespan[-1] * 60

    for route, runtime in routes:
        from_depot_id = line.stations_conn_depots.get(route[0], -1)
        to_depot_id = line.stations_conn_depots.get(route[-1], -1)
        if not line.is_loop_line:
            last_arr_time_first_station = max(arr_slots[(line.line_id, route[0])], timespan[0] * 60)
            # the original desired arrive time at the first station of the route
            desired_arr_time = last_arr_time_first_station + headway
            available_range_arr_time_for_headway = [last_arr_time_first_station + HEADWAY_MIN,
                                                    last_arr_time_first_station + HEADWAY_MAX]
            # the first station of this route does not connect with depots, alter the desired_arr_time
            # for turn-back connections
            potential_from_service_id = -1
            if route[0] not in line.stations_conn_depots:
                # a normal line
                if not line.is_loop_line:
                    is_desired_arr_time_altered = False
                    turnback_sta_id = min(route[0], 2 * len(line.up_stations) - 1 - route[0])
                    front_services = timetable.services_queues[turnback_sta_id]['from']
                    for serv_id in front_services:
                        service = timetable.services[serv_id]
                        if service.next_service == -1:
                            available_range_arr_time_for_turnback = [service.deps[-1] + TURNBACK_MIN_SECS,
                                                                     service.deps[-1] + TURNBACK_MAX_SECS]
                            if is_overlapping(available_range_arr_time_for_headway,
                                              available_range_arr_time_for_turnback):
                                desired_arr_time = overlapping_region(available_range_arr_time_for_headway,
                                                                      available_range_arr_time_for_turnback)[0]
                                potential_from_service_id = service.id
                                is_desired_arr_time_altered = True
                                break
                    # no front services, skip
                    if not is_desired_arr_time_altered:
                        continue

            # if the desired_arr_time exceed the time window
            if desired_arr_time >= end_time_secs:
                continue
            # remain time is not sufficient for this route
            if last_arr_time_first_station + headway + runtime >= END_TIME * 60:
                continue

            # the desired departure time at the last station of the route
            desired_to_time = desired_arr_time + runtime + line.dwells[route[-1]]

            # each terminal station of this route connect with depot
            if from_depot_id != -1 and to_depot_id != -1:
                route_pool.append(route)
                desired_arr_times[(route[0], route[-1])] = desired_arr_time
            # do not have "from" depot
            elif from_depot_id == -1 and to_depot_id != -1:
                route_pool.append(route)
                dedicated_turn_back_servs[(route[0], route[-1])] = ('from', potential_from_service_id)
                desired_arr_times[(route[0], route[-1])] = desired_arr_time
            # do not have "to" depot
            elif to_depot_id == -1 and from_depot_id != -1:
                route_pool.append(route)
                desired_arr_times[(route[0], route[-1])] = desired_arr_time
        # loop line
        else:
            last_arr_time_first_station = max(arr_slots[(line.line_id, route[0])], timespan[0] * 60)
            # the original desired arrive time
            desired_arr_time = last_arr_time_first_station + headway
            # if the desired_arr_time exceed the time window
            if desired_arr_time >= end_time_secs:
                continue
            # remain time is not sufficient for this route
            if last_arr_time_first_station + headway + runtime >= END_TIME * 60:
                continue

            # each terminal station of this route connect with depot
            # the route is valid to be scheduled
            route_pool.append(route)
            desired_arr_times[(route[0], route[-1])] = desired_arr_time

    return route_pool, dedicated_turn_back_servs, desired_arr_times


def construct_service(route, service_nums, dir_id, line, first_arr_time, time_span, timetable, arr_slots):
    # generate service
    service = TrainService(service_nums[line.line_id], dir_id, route, line.line_id)

    arr_time = first_arr_time
    dep_time = arr_time + line.dwells[route[0]]

    service.arrs.append(arr_time)
    service.deps.append(dep_time)

    for i in range(1, len(route)):  # skip the first station of this route
        arr_time = dep_time + line.runtimes[route[i - 1], route[i]]
        dep_time = arr_time + line.dwells[route[i]]
        service.arrs.append(arr_time)
        service.deps.append(dep_time)

    is_valid = True
    if service.arrs[0] >= time_span[-1] * 60 or service.deps[-1] >= END_TIME * 60:
        is_valid = False
    # valid
    else:
        service_nums[line.line_id] += 1
        timetable.services[service.id] = service

        if dir_id == 0:
            timetable.up_services.append(service.id)
        else:
            timetable.dn_services.append(service.id)

        # update the last arrive time (arr_slots)
        for k in range(0, len(service.route) - 1):  # skip the last station
            arr_slots[line.line_id, service.route[k]] = service.arrs[k]
        # the minimum station id and maximum station id of this line in the current direction
        min_sta_id = line.up_stations[0] if service.direction == 0 else line.dn_stations[0]
        max_sta_id = line.up_stations[-1] if service.direction == 0 else line.dn_stations[-1]
        # normal line
        if not line.is_loop_line:
            # complement arr times from "service.route[0] - 1" to min_sta_id
            for sta_id in range(service.route[0] - 1, min_sta_id - 1, -1):
                if sta_id == service.route[0] - 1:
                    arr_slots[line.line_id, sta_id] = service.arrs[0] - line.runtimes[sta_id, sta_id + 1] - line.dwells[
                        sta_id]
                else:
                    arr_slots[line.line_id, sta_id] = arr_slots[line.line_id, sta_id + 1] - \
                                                      line.runtimes[sta_id, sta_id + 1] - line.dwells[sta_id]
            # complement arr times from "service.route[0] + 1" to max_sta_id
            for sta_id in range(service.route[-1] + 1, max_sta_id + 1):
                if sta_id == service.route[-1] + 1:
                    arr_slots[line.line_id, sta_id] = service.arrs[-1] + line.runtimes[sta_id - 1, sta_id] + \
                                                      line.dwells[sta_id - 1]
                else:
                    arr_slots[line.line_id, sta_id] = arr_slots[line.line_id, sta_id - 1] + line.runtimes[
                        sta_id - 1, sta_id] + line.dwells[sta_id - 1]
        # loop line
        else:
            mid_platforms = [i for i in line.stations_conn_depots.keys() if i not in
                             [line.up_stations[0], line.up_stations[-1],
                              line.dn_stations[0], line.dn_stations[-1]]]
            # has middle stations connecting with depots
            if len(mid_platforms) > 0:
                # full-length route, turn-back at the start
                if route[0] == route[-1] and abs(route[0] - route[1]) > 1:
                    mid_sta_id = min(mid_platforms) if service.direction == 0 else max(mid_platforms)
                    arr_t_mid_sta = service.deps[-1] - line.dwells[route[-1]] - line.get_runtime(mid_sta_id, route[-1])
                    arr_slots[line.line_id, arr_t_mid_sta] = arr_t_mid_sta
                # short length route, start from middle platform
                elif route[0] != route[-1] and route[0] in mid_platforms:
                    first_sta_id = route[-1]
                    second_sta_id = line.up_stations[0] if service.direction == 0 else line.dn_stations[0]
                    arr_slots[line.line_id, first_sta_id] = \
                        max(arr_slots[line.line_id, first_sta_id],
                            service.arrs[0] - line.get_runtime(second_sta_id, route[0])
                            - line.runtimes[first_sta_id, second_sta_id])

    return service, is_valid


def is_overlapping(x, y):
    return not (x[1] < y[0] or y[1] < x[0])


def overlapping_region(x, y):
    if is_overlapping(x, y):
        return [max(x[0], y[0]), min(x[1], y[1])]
    else:
        return None


def get_depot_max_flow(depots_dep_times, depots_arr_times, from_depot_id, to_depot_id, from_time, to_time):
    temp_depots_dep_times = copy.deepcopy(depots_dep_times)
    temp_depots_arr_times = copy.deepcopy(depots_arr_times)
    if from_depot_id != -1:
        temp_depots_dep_times[str(from_depot_id)].add(from_time)
    if to_depot_id != -1:
        temp_depots_arr_times[str(to_depot_id)].add(to_time)

    max_depot_flows = {i: 0 for i in [from_depot_id, to_depot_id] if i != -1}

    for depot_id in max_depot_flows.keys():
        max_depot_flow = calculate_depot_max_flow(temp_depots_dep_times, temp_depots_arr_times, depot_id)
        max_depot_flows[depot_id] = max_depot_flow

    return max_depot_flows


def calculate_depot_max_flow(depots_dep_times, depots_arr_times, depot_id):
    current_depot_flow = max_depot_flow = 0
    dep_times = depots_dep_times[str(depot_id)]
    arr_times = depots_arr_times[str(depot_id)]
    i = j = 0
    while i < len(dep_times) or j < len(arr_times):
        # check the current action
        leave_depot = i < len(dep_times) and (j == len(arr_times) or dep_times[i] < arr_times[j])
        enter_depot = j < len(arr_times) and (i == len(dep_times) or dep_times[i] >= arr_times[j])
        # leave depot
        if leave_depot:
            current_depot_flow += 1
            i += 1
        # enter depot
        elif enter_depot:
            current_depot_flow -= 1
            j += 1

        max_depot_flow = max(max_depot_flow, current_depot_flow)
    return max_depot_flow


def delete_services(timetable_net, lines, depots):
    depot_slot_sequence = defaultdict(list)
    for line_id, timetable in timetable_net.items():
        for service in timetable.services.values():
            if service.from_depot_id != -1:
                depot_id = service.from_depot_id
                depot_slot_sequence[service.arrs[0]].append((depot_id, line_id, service.id, 'dep'))
            if service.to_depot_id != -1:
                depot_id = service.to_depot_id
                depot_slot_sequence[service.deps[-1]].append((depot_id, line_id, service.id, 'arr'))

    current_flow = {int(i): 0 for i in depots.keys()}
    depot_slot_sequence = dict(sorted(depot_slot_sequence.items()))
    # 尝试按照客流强度对 sequence 再排序
    for t, sequence in depot_slot_sequence.items():
        for depot_id, line_id, serv_id, action in sequence:
            depot = depots[str(depot_id)]
            if action == 'dep':
                current_flow[depot_id] += 1
            else:
                current_flow[depot_id] -= 1
            if current_flow[depot_id] > depot.capacity:
                # cancel the service, retrieve the flow
                current_flow[depot_id] -= 1
                timetable = timetable_net[line_id]
                service = timetable.services[serv_id]
                # if it has a following service, delete them together
                if service.next_service != -1:
                    # delete the following service from depot_slot_sequence
                    dep_t = timetable.services[service.next_service].arrs[0]
                    arr_t = timetable.services[service.next_service].deps[-1]
                    depot_slot_sequence[dep_t] = [x for x in depot_slot_sequence[dep_t] if x[2] != service.next_service]
                    depot_slot_sequence[arr_t] = [x for x in depot_slot_sequence[arr_t] if x[2] != service.next_service]
                    del timetable.services[service.next_service]
                # if it has a preceding service, delete them together
                if timetable.services[serv_id].front_service != -1:
                    # delete the preceding service from depot_slot_sequence
                    dep_t = timetable.services[service.front_service].arrs[0]
                    arr_t = timetable.services[service.front_service].deps[-1]
                    depot_slot_sequence[dep_t] = [x for x in depot_slot_sequence[dep_t] if
                                                  x[2] != service.front_service]
                    depot_slot_sequence[arr_t] = [x for x in depot_slot_sequence[arr_t] if
                                                  x[2] != service.front_service]
                    del timetable.services[service.front_service]
                del timetable.services[serv_id]
                # update the maximum flow
                depot.maximum_flow = max(depot.maximum_flow, current_flow[depot_id])
