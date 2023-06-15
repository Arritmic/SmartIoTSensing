import numpy as np
import pandas as pd
import time
import os
import sys
from scipy import stats
import math

def get_bit_length_integer(input_value):
    # Compute bit length of an integer
    integer_bit_length = input_value.item().bit_length()
    # print(f"Bit length of integer:", integer_bit_length)

    return integer_bit_length


def get_bit_length_float(input_value):
    # Compute bit length of a float
    float_size_bytes = sys.getsizeof(input_value)
    float_bit_length = float_size_bytes * 8
    # print(f"Bit length of float {input_value} is {float_bit_length} bits")
    return float_bit_length


def get_bit_length_avg_array_int(input_array):
    sum_bit_length = 0
    for i in input_array:
        sum_bit_length += get_bit_length_integer(i)

    return sum_bit_length // len(input_array)


def get_bit_length_avg_array_float(input_array):
    sum_bit_length = 0
    for i in input_array:
        sum_bit_length += get_bit_length_float(i)

    return sum_bit_length // len(input_array)


def compute_confidence_interval_scipy(time_series, confidence_level=0.95):
    """
    Compute the confidence interval of a time series.

    Parameters:
        time_series (ndarray): The time series data.
        confidence (float): The desired confidence level (default: 0.95).

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    n = len(time_series)
    mean = np.mean(time_series)
    std_err = stats.sem(time_series)
    margin_of_error = std_err * stats.t.ppf((1 + confidence_level) / 2, n - 1)
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound


def compute_confidence_interval(time_series, confidence=0.95):
    """
    Compute the confidence interval of a time series.

    Parameters:
        time_series (list): The time series data.
        confidence (float): The desired confidence level (default: 0.95).

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    n = len(time_series)
    mean = sum(time_series) / n
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in time_series) / n)
    margin_of_error = std_dev * 1.96 / math.sqrt(n)  # 1.96 corresponds to a 95% confidence level
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound



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


def main():
    # MANUAL FLAGS. [TODO] Make it automatic or by input arguments
    compute_statistics = True
    compute_bit_length = True

    # LOAD DATA
    data = load_data("../data/group99999.csv")
    print(data)

    temperature = data.loc[data.iloc[:, 2].values == " temp", data.columns[3]]
    temperature_ts = data.loc[data.iloc[:, 2].values == " temp", data.columns[5]]

    air_quality = data.loc[data.iloc[:, 2].values == " quality", data.columns[3]]
    air_quality_ts = data.loc[data.iloc[:, 2].values == " quality", data.columns[5]]

    light = data.loc[data.iloc[:, 2].values == " light", data.columns[3]]
    light_ts = data.loc[data.iloc[:, 2].values == " light", data.columns[5]]

    bitlength_temp = data.loc[data.iloc[:, 2].values == " temp", data.columns[9]]  # The total length of the receive package.

    print(f"[INFO] Size of Temperature values: {len(temperature)}")
    print(f"[INFO] Size of Temperature TS: {len(temperature_ts)}")
    # print(f"[INFO] Size of Temperature Bits: {len(bitlength_temp)}")

    print(f"[INFO] Size of air_quality values: {len(air_quality)}")
    print(f"[INFO] Size of air_quality TS: {len(air_quality_ts)}")

    print(f"[INFO] Size of light values: {len(light)}")
    print(f"[INFO] Size of light TS: {len(light_ts)}")

    if compute_statistics:
        print("\n ********** STATISTICS BENCHMARK: TEMPERATURE   ***************")
        print(f"[TEMP RESULTS] Confidence Interval at 95% of temperature values using float16: {compute_confidence_interval(np.array(temperature).astype(np.float16))}")
        print(f"[TEMP RESULTS] Confidence Interval at 95% of temperature values using float32: {compute_confidence_interval(np.array(temperature).astype(np.float32))}")

        print("\n ********** STATISTICS BENCHMARK: LIGHT   ***************")
        print(f"[LIGHT RESULTS] Confidence Interval at 95% of Light values using float16: {compute_confidence_interval(np.array(light).astype(np.float16))}")
        print(f"[LIGHT RESULTS] Confidence Interval at 95% of Light values using float32: {compute_confidence_interval(np.array(light).astype(np.float32))}")

        print("\n ********** STATISTICS BENCHMARK: AIR QUALITY   ***************")
        print(
            f"[AIR QUALITY RESULTS] Confidence Interval at 95% of Air Quality values using int16: {compute_confidence_interval(np.array(air_quality).astype(np.int16))}")
        print(
            f"[AIR QUALITY RESULTS] Confidence Interval at 95% of Air Quality values using int64: {compute_confidence_interval(np.array(air_quality).astype(np.int64))}")

    if compute_bit_length:
        print("\n ********** BIT LENGTH BENCHMARK: TEMPERATURE   ***************")
        print(f"[TEMP INFO] Average bitlength of temperature values using float16: {get_bit_length_avg_array_float(np.array(temperature).astype(np.float16))}")
        print(f"[TEMP INFO] Average Temperature using float16: {np.mean(np.array(temperature).astype(np.float16))}")
        print(f"[TEMP INFO] Average bitlength of temperature values using float32: {get_bit_length_avg_array_float(np.array(temperature).astype(np.float32))}")
        print(f"[TEMP INFO] Average Temperature using float32: {np.mean(np.array(temperature).astype(np.float32))}")

        print("\n ********** BIT LENGTH BENCHMARK: LIGHT   ***************")
        print(f"[LIGHT INFO] Average bitlength of light values using float16: {get_bit_length_avg_array_float(np.array(light).astype(np.float16))}")
        print(f"[LIGHT INFO] Average Light using float16: {np.mean(np.array(light).astype(np.float16))}")
        print(f"[LIGHT INFO] Average bitlength of light values using float32: {get_bit_length_avg_array_float(np.array(light).astype(np.float32))}")
        print(f"[LIGHT INFO] Average Light using float32: {np.mean(np.array(light).astype(np.float32))}")

        print("\n ********** BIT LENGTH BENCHMARK: AIR QUALITY   ***************")
        print(f"[AIR QUALITY INFO] Average bitlength of air quality values using int8: {get_bit_length_avg_array_int(np.array(air_quality).astype(np.int8))}")
        print(f"[LIGHT INFO] Average Air Quality using int8: {np.mean(np.array(air_quality).astype(np.int8))}")
        print(f"[AIR QUALITY INFO] Average bitlength of air quality values using int16: {get_bit_length_avg_array_int(np.array(air_quality).astype(np.int16))}")
        print(f"[LIGHT INFO] Average Air Quality using int16: {np.mean(np.array(air_quality).astype(np.int16))}")
        print(f"[AIR QUALITY INFO] Average bitlength of air quality values using int64: {get_bit_length_avg_array_int(np.array(air_quality).astype(np.int64))}")
        print(f"[LIGHT INFO] Average Air Quality using int64: {np.mean(np.array(air_quality).astype(np.int64))}")


if __name__ == '__main__':
    print('\n')
    print('*******************************************************************')
    print('*        SmartIoT Data Analysis: Conf Interval  + BitLength       *')
    print('*******************************************************************')
    print('\n')
    main()
