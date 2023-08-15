import math
import os
import csv
import time
import statistics
from RailLine import *
from PassengerFlow import *
from Timetable import *
from TrainService import *
import pandas as pd
import random
import csv
from collections import OrderedDict, defaultdict

INTERVAL = 30  # minutes
START_TIME = 5 * 60  # minutes
END_TIME = 24 * 60
PEAK_HOURS = [(7 * 60, 9 * 60), (17 * 60, 19 * 60)]
FLOW_FACTOR = 1.5
HEADWAY_MIN = 120  # seconds
HEADWAY_MAX = 320
HEADWAY_POOL = [i for i in range(HEADWAY_MIN, HEADWAY_MAX, 100)]
TRAIN_CAPACITY = 1200  # loading capacity
TRANSFER_SECS = 120  # fixed transfer time

# model parameters
TIME_PERIOD = 60  # 1 hour
N = [i for i in range(START_TIME, END_TIME + 1, TIME_PERIOD)]


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

                rail_line.dwells[int(station_id)] = int(dwell_time)
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
        # set down direction dwells
        line.dwells.update({2 * len(line.up_stations) - sta_id - 1: value for sta_id, value in line.dwells.items()})

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
                # save transfer connections
                temp_key = (int(line_id1), station_id1, int(line_id2), station_id2)
                if temp_key not in line1.transfer_pairs:
                    line1.transfer_pairs.append(temp_key)


def gen_timetables(iter_max, lines, passenger_flows):
    root_folder = './test/'
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)

    random.seed(2023)

    timetable_pool = []
    wait_time_dict = {}  # key l,n,i
    weights_dict = {(l, n): [100 for i in HEADWAY_POOL] for l in lines.keys() for n in N}  # key l,n
    for i in range(0, iter_max):
        print("iter: " + str(i))
        # generate a timetable for the network
        timetable_net = {l: Timetable(l) for l in lines.keys()}  # key: line_id

        arr_slots = {}  # key:(l,sta_id), the last arrive time of each station
        for l, line in lines.items():
            for sta_id in line.up_stations + line.dn_stations:
                arr_slots[l, sta_id] = 0

        service_nums = {l: 0 for l in lines.keys()}

        # for each time window
        for n in N:
            is_in_peak_hour = True if (PEAK_HOURS[0][0] <= n < PEAK_HOURS[0][1]) or (
                    PEAK_HOURS[1][0] <= n < PEAK_HOURS[1][1]) else False
            time_span = [k for k in range(n, n + TIME_PERIOD)]
            for l, line in lines.items():  # for each line
                # select a fixed headway for this line within this window
                headway = roulette_selection(HEADWAY_POOL, weights_dict[l, n])
                timetable = timetable_net[l]

                start_time_secs = time_span[0] * 60
                need_break = False
                while not need_break:
                    for route in line.routes:
                        # not in a peak hour, do not schedule short route services
                        if (not is_in_peak_hour) and len(route) < len(line.up_stations):
                            continue
                        direction = 0 if route[0] in line.up_stations else 1
                        service = TrainService(service_nums[l], direction, route)

                        arr_time = max(start_time_secs, arr_slots[l, route[0]] + headway)
                        dep_time = arr_time + line.dwells[route[0]]

                        service.arrs.append(arr_time)
                        service.deps.append(dep_time)

                        for sta_id in route[1:]:  # skip the first station of this route
                            arr_time = dep_time + line.runtimes[sta_id - 1, sta_id]
                            dep_time = arr_time + line.dwells[sta_id]
                            service.arrs.append(arr_time)
                            service.deps.append(dep_time)

                        if service.arrs[0] >= time_span[-1] * 60 or service.deps[-1] >= END_TIME * 60:
                            need_break = True
                            break

                        service_nums[l] += 1
                        timetable.services[service.id] = service

                        if direction == 0:
                            timetable.up_services.append(service.id)
                        else:
                            timetable.dn_services.append(service.id)

                        # update the last arrive time (arr_slots)
                        for k in range(0, len(service.route)):
                            sta_id = service.route[k]
                            arr_slots[l, sta_id] = service.arrs[k]
                        min_sta_id = line.up_stations[0] if service.direction == 0 else line.dn_stations[0]
                        max_sta_id = line.up_stations[-1] if service.direction == 0 else line.dn_stations[-1]
                        for sta_id in range(service.route[0] - 1, min_sta_id - 1, -1):
                            if sta_id == service.route[0] - 1:
                                arr_slots[l, sta_id] = service.arrs[0] - line.runtimes[sta_id, sta_id + 1] - \
                                                       line.dwells[sta_id]
                            else:
                                arr_slots[l, sta_id] = arr_slots[l, sta_id + 1] - line.runtimes[
                                    sta_id, sta_id + 1] - line.dwells[sta_id]
                        for sta_id in range(service.route[-1] + 1, max_sta_id + 1):
                            if sta_id == service.route[-1] + 1:
                                arr_slots[l, sta_id] = service.arrs[-1] + line.runtimes[sta_id - 1, sta_id] + \
                                                       line.dwells[sta_id - 1]
                            else:
                                arr_slots[l, sta_id] = arr_slots[l, sta_id - 1] + line.runtimes[
                                    sta_id - 1, sta_id] + line.dwells[sta_id - 1]

        timetable_pool.append(timetable_net)

        start_time = time.time()
        passenger_flow_simulate(timetable_net, lines, passenger_flows)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"程序执行时间: {execution_time:.4f} 秒")

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
        folder_name = folder + str(l) + '/'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        for serv_id, service in timetable.services.items():
            csv_file_path = folder_name + 'vehicle[{}].csv'.format(str(serv_id))
            csv_data = [['line_idx', 'sta_idx', 'dir', 'type', 'time', 'depot_idx']]
            for i in range(0, len(service.route)):
                sta_id = service.route[i]
                arr_time = service.arrs[i]
                dep_time = service.deps[i]
                direction = service.direction
                if direction == 1:
                    sta_id = 2 * len(line.up_stations) - sta_id - 1
                csv_data.append([l, sta_id, direction, 'arr', arr_time, -1])
                csv_data.append([l, sta_id, direction, 'dep', dep_time, -1])

            csv_data.insert(1, [-1, -1, -1, 'depot_out', service.arrs[0] - 120, 0])
            csv_data.append([-1, -1, -1, 'depot_in', service.deps[-1] + 120, 0])

            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(csv_data)


def passenger_flow_simulate(timetable_net, lines, passenger_flows):
    arrive_slots = defaultdict(list)
    depart_slots = defaultdict(list)
    core_slots = defaultdict(list)

    # !!! t (seconds) is starting from START_TIME
    alight_volume = {}  # key (l,s,t)
    transfer_volume = {}  # key (l2,s2)
    transfer_volume_points = {}  # key (l2,s2)

    # set core_slots, arrive_slots and depart_slots
    for l, timetable in timetable_net.items():
        for service_id, service in timetable.services.items():
            for t in service.arrs:
                arrive_slots[t].append((l, service_id))
                if 'arr' not in core_slots[t]:
                    core_slots[t].append('arr')
            for t in service.deps:
                depart_slots[t].append((l, service_id))
                if 'dep' not in core_slots[t]:
                    core_slots[t].append('dep')
    # sort
    arrive_slots = OrderedDict(sorted(arrive_slots.items()))
    depart_slots = OrderedDict(sorted(depart_slots.items()))
    core_slots = OrderedDict(sorted(core_slots.items()))

    # initialize p_wait, p_train, p_alight, p_transfer
    for l, timetable in timetable_net.items():
        timetable.p_wait = {sta_id: [(START_TIME * 60, 0)] for sta_id in (lines[l].up_stations + lines[l].dn_stations)}
        timetable.p_train = {serv_id: {sta_id: 0 for sta_id in serv.route} for serv_id, serv in
                             timetable.services.items()}
        timetable.p_alight = {serv_id: {sta_id: 0 for sta_id in serv.route} for serv_id, serv in
                              timetable.services.items()}
        timetable.p_transfer = {serv_id: {(s1, s2): 0 for (_, s1, _, s2) in lines[l].transfer_pairs}
                                for serv_id in timetable.services.keys()}

    # simulate passenger flows
    for t, action_list in core_slots.items():  # t is starting from START_TIME
        t_minute = round((int(t) - START_TIME * 60) / 60)
        # this slot contains arrivals
        if 'arr' in action_list:
            # calculate the alight passengers and transfer passengers
            for l, service_id in arrive_slots[t]:
                timetable = timetable_net[l]
                service = timetable.services[service_id]
                station_id = service.route[service.arrs.index(t)]
                alight_rate = passenger_flows.theta[int(l), station_id, t_minute]
                passenger_in_train = 0 if station_id == service.route[0] else timetable.p_train[service_id][
                    station_id - 1]
                alight_pas = math.floor(passenger_in_train * alight_rate)
                # update alight passengers
                timetable.p_alight[service_id][station_id] = alight_pas
                if alight_pas > 0:
                    alight_volume[l, station_id, t] = alight_pas
                # update transfer passengers
                for (l1, s1, l2, s2) in lines[l].transfer_pairs:
                    transfer_rate = passenger_flows.phi[l1, s1, l2, s2, t_minute]
                    timetable.p_transfer[service_id][s1, l2, s2] = alight_pas * transfer_rate
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
                pre_time_second, remaining_pas = timetable.p_wait[station_id][-1]
                pre_time_minute = round((pre_time_second - START_TIME * 60) / 60)
                arriving_pas = math.ceil(sum([passenger_flows.d[int(l), station_id, t_slot]
                                              for t_slot in range(pre_time_minute, t_minute)]))
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

                wait_passengers = remaining_pas + arriving_pas + transfer_pas
                timetable.p_wait[station_id].append((t, wait_passengers))

        # this slot contains departures
        if 'dep' in action_list:
            # calculate the waiting passengers and boarding passengers
            for l, service_id in depart_slots[t]:
                timetable = timetable_net[l]
                service = timetable.services[service_id]
                station_id = service.route[service.deps.index(t)]
                # remaining passengers + new arriving passengers + transfer passengers
                pre_time_second, remaining_pas = timetable.p_wait[station_id][-1]
                pre_time_minute = round((pre_time_second - START_TIME * 60) / 60)
                arriving_pas = sum([passenger_flows.d[int(l), station_id, t_slot]
                                    for t_slot in range(pre_time_minute, t_minute)])
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

                wait_passengers = remaining_pas + arriving_pas + transfer_pas
                timetable.p_wait[station_id].append((t, wait_passengers))
                passenger_in_train = 0 if station_id == service.route[0] else timetable.p_train[service_id][
                    station_id - 1]
                boarding_passengers = max(wait_passengers, TRAIN_CAPACITY - passenger_in_train)
                # update en-route passengers
                timetable.p_train[service_id][station_id] = passenger_in_train + boarding_passengers
                # update remaining passengers
                wait_passengers -= boarding_passengers
                timetable.p_wait[station_id].append((t, wait_passengers))
