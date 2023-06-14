import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import matplotlib.dates as mdates

Pr = 1
mean_temp_global = 0


def get_interpolated_array(input_array, interp=1):
    input_array = input_array[::interp]
    return input_array


def get_cost_processing(used_time):
    cost = 10 * used_time * Pr
    # print(f"    >> [COST] Processing cost = {cost * 1000} mJ")
    return cost * 1000


def get_cost_storage(input_array, iterations=100, interp=1):
    # input_array = np.ones(3600, dtype=float)*22.7
    input_array = get_interpolated_array(input_array, interp=interp)
    new_array = []
    array_times = []
    for i in range(iterations):
        total_time = 0
        for value in input_array:
            t0 = time.time()
            new_array.append(value)
            total_time += time.time() - t0
        array_times.append(total_time)
        new_array = []

    cost = get_cost_processing(np.mean(array_times))
    return cost


def get_cost_average(input_array, iterations=100, interp=1):
    # input_array = np.ones(3600, dtype=float)*22.7
    input_array = get_interpolated_array(input_array, interp=interp)
    array_times = []
    averages = []
    for i in range(iterations):
        total_time = 0
        average = input_array[0]
        for value in input_array:
            t0 = time.time()
            average = (average + value) / 2
            total_time += time.time() - t0
        array_times.append(total_time)
        averages.append(average)
    cost = get_cost_processing(np.mean(array_times))
    return cost, np.mean(averages)


def load_data(file_path):
    """ Check if csv file with data exists"""

    pd.set_option("display.precision", 15)
    print(f"    >> Reading data from: {file_path}")

    stored = False
    if os.path.isfile(file_path):
        stored = True
        data = pd.read_csv(file_path, index_col=False, dtype='str')
        # data = data.astype('float64')

    return data


def plot_cost_energy(input_values_storage, cost_average, input_sizes):
    print(len(cost_average))
    print(cost_average)
    plt.plot(input_sizes, input_values_storage, 'r', input_sizes, cost_average, 'g')
    plt.grid(True, which='major', color='#666666', linestyle='-')
    plt.legend(['Energy cost storage (mJ)', 'Energy cost average (mJ)'], loc='upper right', fontsize=14)
    plt.minorticks_on()
    plt.grid(True, which='minor', color='#999999', linestyle='dotted', alpha=0.3)
    plt.xlabel('Number of floats', fontsize=14)
    plt.ylabel('Energy (mJ)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Energy consumption data for temperature values in RPi3 for group99999", fontsize=20)
    # plt.show()


def plot_sensor_data(input_values, input_ts, mean_temps):

    fig = plt.figure(figsize=(10, 8))
    locator = mdates.HourLocator(interval=1)
    locator.MAXTICKS = 15000

    plt.subplot(4, 1, 1)
    plt.plot(input_ts[0], input_values[0], 'r')
    plt.grid(True, which='major', color='#666666', linestyle='-')
    plt.legend(['Temperature (°C)'], loc='upper right', fontsize=14)
    plt.minorticks_on()
    plt.grid(True, which='minor', color='#999999', linestyle='dotted', alpha=0.3)
    # plt.xlabel('Timestamp', fontsize=14)
    plt.ylabel('Temperature', fontsize=14)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Temperature sensor data from RPi3 for group99999", fontsize=20)

    plt.subplot(4, 1, 2)
    plt.plot(input_ts[1], input_values[1], 'g')
    plt.grid(True, which='major', color='#666666', linestyle='-')
    plt.legend(['Air Quality (CO2 Levels)'], loc='upper right', fontsize=14)
    plt.minorticks_on()
    plt.grid(True, which='minor', color='#999999', linestyle='dotted', alpha=0.3)
    plt.xlabel('Timestamp', fontsize=14)
    plt.ylabel('Air Quality (CO2)', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.ylim(np., y_max)  # Set the y-axis limits
    # print(input_values[1])
    plt.yticks(np.arange(np.min(input_values[1]), np.max(input_values[1]) + 1, 400), fontsize=12)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.subplot(4, 1, 3)
    plt.plot(input_ts[2], input_values[2], 'm')
    plt.grid(True, which='major', color='#666666', linestyle='-')
    plt.legend(['Light intensity (Lux)'], loc='upper right', fontsize=14)
    plt.minorticks_on()
    plt.grid(True, which='minor', color='#999999', linestyle='dotted', alpha=0.3)
    plt.xlabel('Timestamp', fontsize=14)
    plt.ylabel('Light intensity (Lux)', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.ylim(np., y_max)  # Set the y-axis limits
    # print(input_values[1])
    plt.yticks(np.arange(np.min(input_values[2]), np.max(input_values[2]) + 1, 200), fontsize=12)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.subplot(4, 1, 4)
    plt.plot(input_ts[0], input_values[3], 'c')
    plt.grid(True, which='major', color='#666666', linestyle='-')
    plt.legend(['Package length transmitted'], loc='upper right', fontsize=14)
    plt.minorticks_on()
    plt.grid(True, which='minor', color='#999999', linestyle='dotted', alpha=0.3)
    plt.xlabel('Timestamp', fontsize=14)
    plt.ylabel('Package Length (bits)', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.ylim(np., y_max)  # Set the y-axis limits
    # print(input_values[1])
    # plt.yticks(np.arange(np.min(input_values[2]), np.max(input_values[2]) + 1, 100), fontsize=12)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.show()


def main():
    data = load_data("../data/group99999.csv")
    print(data)

    temperature = data.loc[data.iloc[:, 2].values == " temp", data.columns[3]]
    temperature_ts = data.loc[data.iloc[:, 2].values == " temp", data.columns[5]]

    air_quality = data.loc[data.iloc[:, 2].values == " quality", data.columns[3]]
    air_quality_ts = data.loc[data.iloc[:, 2].values == " quality", data.columns[5]]

    light = data.loc[data.iloc[:, 2].values == " light", data.columns[3]]
    light_ts = data.loc[data.iloc[:, 2].values == " light", data.columns[5]]

    bitlength_temp = data.loc[data.iloc[:, 2].values == " temp", data.columns[9]]

    print(f"[INFO] Size of Temperature values: {len(temperature)}")
    print(f"[INFO] Size of Temperature TS: {len(temperature_ts)}")
    print(f"[INFO] Size of Temperature Bits: {len(bitlength_temp)}")

    print(f"[INFO] Size of air_quality values: {len(air_quality)}")
    print(f"[INFO] Size of air_quality TS: {len(air_quality_ts)}")

    print(f"[INFO] Size of light values: {len(light)}")
    print(f"[INFO] Size of light TS: {len(light_ts)}")

    data_container = []
    ts_container = []
    data_container.append(np.array(temperature).astype(float))
    data_container.append(np.array(air_quality).astype(int))
    data_container.append(np.array(light).astype(float))
    data_container.append(np.array(bitlength_temp).astype(int))

    ts_container.append(temperature_ts)
    ts_container.append(air_quality_ts)
    ts_container.append(light_ts)

    energy_cost_storage = []
    energy_cost_averaging = []
    energy_cost_number = []
    interpolations = [1, 2, 3, 4, 5, 6, 8, 10]
    mean_temp_by_amount = []

    # simulation_amount = 3600
    for i in interpolations:
        cost = get_cost_storage(np.array(temperature).astype(float), interp=i)
        cost_avg, mean_temp_avg = get_cost_average(np.array(temperature).astype(float), interp=i)
        print(f"[RESULTS] Cost temperature storage for {len(temperature) // i} values: {cost}  mJ")
        print(f"[RESULTS] Cost temperature average for {len(temperature) // i} values: {cost_avg}  mJ")
        print(f"[RESULTS] Mean temperature for {len(temperature) // i} values: {mean_temp_avg} °C")
        energy_cost_storage.append(cost)
        energy_cost_averaging.append(cost_avg)
        energy_cost_number.append(len(temperature) // i)
        mean_temp_by_amount.append(mean_temp_avg)
        # energy_cost_number.append(simulation_amount // i)

    print("\n")
    print(f"[RESULTS] GT Mean temperature for {len(temperature)} values: {np.mean(np.array(temperature).astype(float))} °C")
    times_mean_comp_temp = []
    for i in interpolations:
        temperatures_vector_interpolate = np.array(temperature).astype(float)
        temperatures_vector_interpolate = temperatures_vector_interpolate[::i]
        t0 = time.time()
        mean_temp = np.mean(temperatures_vector_interpolate)
        tf = time.time() - t0
        times_mean_comp_temp.append(tf)
        print(f"[RESULTS] ES Mean temperature for {len(temperature)//i} values: {mean_temp}  C")

    energy_cost_storage = np.array(energy_cost_storage)
    times_mean_comp_temp = np.array(times_mean_comp_temp)
    energy_cost_storage = energy_cost_storage + times_mean_comp_temp
    plot_cost_energy(energy_cost_storage, energy_cost_averaging, energy_cost_number)
    plot_sensor_data(data_container, ts_container, mean_temp_avg)


if __name__ == '__main__':
    print('\n')
    print('*******************************************************')
    print('*         SmartIoT Data Analysis: Group99999          *')
    print('*******************************************************')
    print('\n')
    main()
