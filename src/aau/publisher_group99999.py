import time
import paho.mqtt.client as mqtt
import util
import sensor_api as sapi
import smbus
import numpy as np



# VARIABLES
temperature_array = []




def compute_temperature_parameters(input_array):
  max_temp = np.max(input_array)
  mean_temp = np.mean(input_array)
  return max_temp, mean_temp


# The callback for when the client receives a CONNACK response from the server.


client = mqtt.Client()
client.on_connect = util.on_connect
client.on_message = util.on_message


print(util.server)
client.connect(util.server,1883,60)



# intialize sensors, so we can get data out 
bus = smbus.SMBus(1)  # Rev 2 Pi uses 1
sensor = sapi.BH1750(bus)
bme680_sensor = sapi.sensor_bme680()  # Temperature, and Humidity sensor
sgp30_sensor = sapi.sensor_sgp30() # Air quality sensor

######################### Logic for furter programming #######################

counter = 0
arr_size = 60
userid=util.userid
extra_counter=0

# Temperature, pressure and humidty - BME680
data_temp = [0] * arr_size
timestamps_temp = [0] * arr_size

data_pressure = [0] * arr_size
timestamps_pressure = [0] * arr_size


data_humidity = [0] * arr_size
timestamps_humidity = [0] * arr_size

# Air quality - SGP30
data_airquality = [0] * arr_size
timestamps_airquality = [0] * arr_size

# Light - BH1750
data_lowres = [0]*arr_size
timestamps_lowres = [0]*arr_size

data_highres = [0]*arr_size
timestamps_highres = [0]*arr_size
data_highres2 = [0]*arr_size
timestamps_highres2 = [0]*arr_size


while(1):
    time.sleep(2)

    #Temperature sample in celcius
    temp_sample, ts_temp = bme680_sensor.get_temp()
    #print(temp_sample, ts_temp)
    data_temp[counter] = temp_sample
    timestamps_temp[counter] = ts_temp
    #Humidity sample
    humidity_sample, ts_humid = bme680_sensor.get_humidity()
    data_humidity[counter] = humidity_sample
    timestamps_humidity[counter] = ts_humid
    #Pressure sample
    pressure_sample, ts_pressure = bme680_sensor.get_pressure()
    data_pressure[counter] = pressure_sample
    timestamps_pressure[counter] = ts_pressure

    #airquality (CO2)
    airqual_sample, ts_qual = sgp30_sensor.get_sample()
    data_airquality[counter] = airqual_sample
    timestamps_airquality[counter] = ts_qual

    #Light samples Lx
    sample,ts = sensor.measure_low_res()
    data_lowres[counter] = sample
    timestamps_lowres[counter] = ts

    sample, ts = sensor.measure_high_res()
    data_highres[counter] = sample
    timestamps_highres[counter] = ts
    sample, ts = sensor.measure_high_res2()
    data_highres2[counter] = sample
    timestamps_highres2[counter] = ts
    sensor.set_sensitivity((sensor.mtreg + 10) % 255)

    counter +=1
    temperature_array.append(temp_sample)

    if counter == arr_size:
	max_temp, mean_temp = compute_temperature_parameters(temperature_array)
 	print("   >> Max temperature = " + str(max_temp))
 	print("   >> Average temperature = " + str(mean_temp))

	temperature_array = []
        extra_counter+=1
        counter = 0

        data=util.prepare_payload(["temp", "quality"], [data_temp, data_airquality],
                                   [timestamps_temp, timestamps_airquality])
        util.send_topics(data,userid,client)
        data_light = [0] * arr_size
        timestamps_light = [0] * arr_size

        data_temp = [0] * arr_size
        timestamps_temp = [0] * arr_size

        data_pressure = [0] * arr_size
        timestamps_pressure = [0] * arr_size

        data_airquality = [0] * arr_size
        timestamps_airquality = [0] * arr_size

        data_humidity = [0] * arr_size
        timestamps_humidity = [0] * arr_size
        if extra_counter == 10:
            exit()

        
