import numpy as np
import time
import paho.mqtt.client as mqtt
import util
import sensor_api as sapi
import smbus2
import sys
import math


def timerGroup99999(f):
    def wrapper(*args, **kw):
        t_start = time.time()
        result = f(*args, **kw)
        t_end = time.time()
        return result, t_end - t_start

    return wrapper


@timerGroup99999
def compute_confidence_interval(time_series, confidence=0.95):
    # Compute the confidence interval of a time series
    n = len(time_series)
    mean = sum(time_series) / n
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in time_series) / n)
    margin_of_error = std_dev * 1.96 / math.sqrt(n)  # 1.96 corresponds to a 95% confidence level
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound


@timerGroup99999
def artificial_light(input_data_light, counter, artificial_light_threshold=200):
    # Function to determine if the artificial light is on based on sensed values
    few_sensed_values = input_data_light[(counter - 10):counter]
    d = 0
    light_is_on = False
    light_on = few_sensed_values > artificial_light_threshold
    if sum(light_on) > len(light_on) / 2:
        light_is_on = True
    return light_is_on


@timerGroup99999
def artificial_air(input_data_air, counter, artificial_air_threshold=50):
    # Function to determine if the artificial light is on based on sensed values
    few_sensed_values = input_data_air[(counter - 10):counter]
    d = 0
    air_is_on = False
    air_on = few_sensed_values > artificial_air_threshold
    if sum(air_on) > len(air_on) / 2:
        air_is_on = True
    return air_is_on


@timerGroup99999
def get_statistics_float(time_series, type_):
    # Function to compute statistics (average, confidence interval, peak) of a time series
    average = np.average(time_series)
    [up_quantile, low_quantile] = compute_confidence_interval(time_series)
    peak = max(time_series) if type_ == "max" else min(time_series)
    return np.float16(average), np.float16(up_quantile), np.float16(low_quantile), np.float16(peak)

@timerGroup99999
def get_statistics_integer(time_series, type_):
    # Function to compute statistics (average, confidence interval, peak) of a time series
    average = np.average(time_series)
    [up_quantile, low_quantile] = compute_confidence_interval(time_series)
    peak = max(time_series) if type_ == "max" else min(time_series)
    return np.int16(average), np.int16(up_quantile), np.int16(low_quantile), np.int16(peak)


# MQTT client initialization
client = mqtt.Client()
client.on_connect = util.on_connect
client.on_message = util.on_message

print(util.server)
client.connect(util.server, 1883, 60)

bus = smbus2.SMBus(1)
sensor = sapi.BH1750(bus)
bme680_sensor = sapi.sensor_bme680()
sgp30_sensor = sapi.sensor_sgp30()

counter = 0
arr_size = 5
userid = util.userid
extra_counter = 0

# Initialize arrays for storing sensor data and timestamps
data_temp = [0] * arr_size
timestamps_temp = [0] * arr_size
data_pressure = [0] * arr_size
timestamps_pressure = [0] * arr_size
data_humidity = [0] * arr_size
timestamps_humidity = [0] * arr_size
data_airquality = [0] * arr_size
timestamps_airquality = [0] * arr_size
data_lowres = [0] * arr_size
timestamps_lowres = [0] * arr_size
data_highres = [0] * arr_size
timestamps_highres = [0] * arr_size
data_highres2 = [0] * arr_size
timestamps_highres2 = [0] * arr_size
data_temperature = np.zeros(3600, dtype=np.float16)
data_air_quality = np.zeros(3600, dtype=np.int16)
data_light = np.zeros(3600, dtype=np.float16)
timestamp_temperature = 0
timestamp_airquality = 0
timestamp_light = 0
Change_Air = False
Change_Light = False
Is_Light_On_Now = False
Is_Air_Open_Now = False

while True:
    time.sleep(1)
    temp_sample, ts_temp = bme680_sensor.get_temp()
    data_temperature[counter] = temp_sample
    timestamp_temperature = ts_temp
    humidity_sample, ts_humid = bme680_sensor.get_humidity()
    pressure_sample, ts_pressure = bme680_sensor.get_pressure()
    airqual_sample, ts_qual = sgp30_sensor.get_sample()
    data_air_quality[counter] = airqual_sample
    timestamp_airquality = ts_qual
    sample, ts = sensor.measure_low_res()
    sample, ts = sensor.measure_high_res()
    timestamp_light = ts
    data_light[counter] = sample
    counter += 1

    if counter == 10:
        # [TODO] For what is this? Cannot it be an initialization?
        Is_Light_On = artificial_light(input_data_light=data_light, counter=counter, artificial_light_threshold=200)
        Is_Air_Open = artificial_air(input_data_air=data_air_quality, counter=counter, artificial_air_threshold=50)
        Change_Air = True
        Change_Light = True
        print(Is_Air_Open, Is_Light_On)

    if counter > 10:
        Is_Light_On = artificial_light(input_data_light=data_light, counter=counter, artificial_light_threshold=200)
        Is_Air_Open = artificial_air(input_data_air=data_air_quality, counter=counter, artificial_air_threshold=50)
        if Is_Light_On_Now != Is_Light_On:
            Change_Light = True
        if Is_Air_Open_Now != Is_Air_Open:
            Change_Air = True

    interruption_trigger = Change_Light or Change_Air
    Is_Light_On = Is_Light_On_Now
    Is_Air_Open = Is_Air_Open_Now
    print(counter)

    if counter == 10:
        counter = 0
        print("inside")
        for measurement_type in [1, 2, 3]:
            if measurement_type == 1:
                print("1")
                [temp_avg, temp_up, temp_down, temp_max], time_this_function = get_statistics_float(data_temperature, "max")
                timestamp_now = timestamp_temperature
                measurement_names = ["temp_avg", "temp_up", "temp_down", "temp_max"]
                measurement_data = [[temp_avg], [temp_up], [temp_down], [temp_max]]
                measurement_timestamps = [[timestamp_now], [timestamp_now], [timestamp_now], [timestamp_now]]
                data = util.prepare_payload(measurement_names, measurement_data, measurement_timestamps)
                data_temperature = np.zeros(3600, dtype=np.float16)
            elif measurement_type == 2:
                print("2")
                [light_avg, light_up, light_down, light_max], time_this_function = get_statistics_float(data_light, "min")
                timestamp_now = timestamp_light
                measurement_names = ["light_avg", "light_up", "light_down", "light_max"]
                measurement_data = [[light_avg], [light_up], [light_down], [light_max]]
                measurement_timestamps = [[timestamp_now], [timestamp_now], [timestamp_now], [timestamp_now]]
                data = util.prepare_payload(measurement_names, measurement_data, measurement_timestamps)
                data_light = np.zeros(3600, dtype=np.float16)
            else:
                print("3")
                [air_quality_avg, air_quality_up, air_quality_down, air_quality_min], time_this_function = get_statistics_integer(data_air_quality, "min")
                timestamp_now = timestamp_airquality
                measurement_names = ["air_quality_avg", "air_quality_up", "air_quality_down", "air_quality_min"]
                measurement_data = [[air_quality_avg], [air_quality_up], [air_quality_down], [air_quality_min]]
                measurement_timestamps = [[timestamp_now], [timestamp_now], [timestamp_now], [timestamp_now]]
                data = util.prepare_payload(measurement_names, measurement_data, measurement_timestamps)
                data_air_quality = np.zeros(3600, dtype=np.int16)
            util.send_topics(data, userid, client)

    if interruption_trigger:
        print("int trigg")
        if Change_Light:
            timestamp_now = timestamp_light
            measurement_names = ["is_light_on", "red"]
            if Is_Light_On:
                measurement_data = [[1], [1]]
            else:
                measurement_data = [[0], [0]]
            measurement_timestamps = [[timestamp_now], [timestamp_now]]
            data = util.prepare_payload(measurement_names, measurement_data, measurement_timestamps)
            util.send_topics(data, userid, client)
        if Change_Air:
            timestamp_now = timestamp_airquality
            measurement_names = ["is_window_open", "red"]
            if Is_Air_Open:
                measurement_data = [[1], [1]]
            else:
                measurement_data = [[0], [0]]
            measurement_timestamps = [[timestamp_now], [timestamp_now]]
            data = util.prepare_payload(measurement_names, measurement_data, measurement_timestamps)
            util.send_topics(data, userid, client)
        Change_Light = False
        Change_Air = False
