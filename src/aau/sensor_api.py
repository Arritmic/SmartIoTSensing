from datetime import datetime
import random
import bme680
from sgp30 import SGP30
import time
import sys


# Sensor APIs, using predefined function calls to acquire sensor data.


def get_timestamp():
    now=datetime.now()
    sample_ts=now.strftime("%H:%M:%S")
    return sample_ts


class sensor_sgp30():
    def __init__(self):
        super().__init__()
        self.sgp30 = SGP30()
        self.sgp30.start_measurement(self.crude_progress_bar)
        sys.stdout.write('\n')
    
    def crude_progress_bar(self):
        sys.stdout.write('.')
        sys.stdout.flush()

    def get_sample(self):
        timestamp=get_timestamp()
        CO2_result=[]
        total_voc_result=[]
        out = str(self.sgp30.get_air_quality())
        vals = out.rsplit("\n")
        CO2 = vals[1].rsplit(" ")
        for x in CO2:
            if x != "":
                CO2_result.append(x)
        CO2_result.pop(0)
        CO2_result.pop(0)


        total_voc= vals[2].rsplit(" ")
        for x in total_voc:
            if x != "":
                total_voc_result.append(x)
        total_voc_result.pop(0)
        total_voc_result.pop(0)
        CO2_out = int(CO2_result[0]) #+ " "+ CO2_result[1]
        total_voc_out = int(total_voc_result[0]) #+ " " + total_voc_result[1]

        #print("CO2:", CO2_out, "total_voc:" , total_voc_out)
        return CO2_out, timestamp
        #return CO2_out, total_voc_out, 





class sensor_bme680():
    def __init__(self):
        super().__init__()
        try:
            self.sensor = bme680.BME680(bme680.I2C_ADDR_PRIMARY)
        except (RuntimeError, IOError):
            self.sensor = bme680.BME680(bme680.I2C_ADDR_SECONDARY)

        # These calibration data can safely be commented
        # out, if desired.

        print('Calibration data:')
        for name in dir(self.sensor.calibration_data):

            if not name.startswith('_'):
                value = getattr(self.sensor.calibration_data, name)

                if isinstance(value, int):
                    print('{}: {}'.format(name, value))

        # These oversampling settings can be tweaked to
        # change the balance between accuracy and noise in
        # the data.

        self.sensor.set_humidity_oversample(bme680.OS_2X)
        self.sensor.set_pressure_oversample(bme680.OS_4X)
        self.sensor.set_temperature_oversample(bme680.OS_8X)
        self.sensor.set_filter(bme680.FILTER_SIZE_3)
        self.sensor.set_gas_status(bme680.ENABLE_GAS_MEAS)

        print('\n\nInitial reading:')
        for name in dir(self.sensor.data):
            value = getattr(self.sensor.data, name)

            if not name.startswith('_'):
                print('{}: {}'.format(name, value))

        self.sensor.set_gas_heater_temperature(320)
        self.sensor.set_gas_heater_duration(150)
        self.sensor.select_gas_heater_profile(0)

        return
    def get__all(self):
        if self.sensor.get_sensor_data():
            output = '{0:.2f} C,{1:.2f} hPa,{2:.2f} %RH'.format(
                self.sensor.data.temperature,
                self.sensor.data.pressure,
                self.sensor.data.humidity)
                

            if self.sensor.data.heat_stable:
                print('{0},{1} Ohms'.format(
                    output,
                    self.sensor.data.gas_resistance))
            else:
                print(output)
            return output

    def get_temp(self):
        timestamp = get_timestamp()
        sens_val= self.sensor.data.temperature

        #print(sens_val , "C")
        return sens_val, timestamp

    def get_pressure(self):
        timestamp = get_timestamp()

        sens_val = self.sensor.data.pressure
        #print(sens_val, " hPA")
        return sens_val, timestamp

    def get_humidity(self):
        timestamp = get_timestamp()

        sens_val = self.sensor.data.humidity
        #print(sens_val, "%RH")
        return sens_val, timestamp


class BH1750():
    """ Implement BH1750 communication. """
    # Define some constants from the datasheet
    POWER_DOWN = 0x00 # No active state
    POWER_ON   = 0x01 # Power on
    RESET      = 0x07 # Reset data register value
    # Start measurement at 4lx resolution. Time typically 16ms.
    CONTINUOUS_LOW_RES_MODE = 0x13
    # Start measurement at 1lx resolution. Time typically 120ms
    CONTINUOUS_HIGH_RES_MODE_1 = 0x10
    # Start measurement at 0.5lx resolution. Time typically 120ms
    CONTINUOUS_HIGH_RES_MODE_2 = 0x11
    # Start measurement at 1lx resolution. Time typically 120ms
    # Device is automatically set to Power Down after measurement.
    ONE_TIME_HIGH_RES_MODE_1 = 0x20
    # Start measurement at 0.5lx resolution. Time typically 120ms
    # Device is automatically set to Power Down after measurement.
    ONE_TIME_HIGH_RES_MODE_2 = 0x21
    # Start measurement at 1lx resolution. Time typically 120ms
    # Device is automatically set to Power Down after measurement.
    ONE_TIME_LOW_RES_MODE = 0x23

    def __init__(self, bus, addr=0x23):
        self.bus = bus
        self.addr = addr
        self.power_down()
        self.set_sensitivity()

    def _set_mode(self, mode):
        self.mode = mode
        self.bus.write_byte(self.addr, self.mode)

    def power_down(self):
        self._set_mode(self.POWER_DOWN)

    def power_on(self):
        self._set_mode(self.POWER_ON)

    def reset(self):
        self.power_on() #It has to be powered on before resetting
        self._set_mode(self.RESET)

    def cont_low_res(self):
        self._set_mode(self.CONTINUOUS_LOW_RES_MODE)

    def cont_high_res(self):
        self._set_mode(self.CONTINUOUS_HIGH_RES_MODE_1)

    def cont_high_res2(self):
        self._set_mode(self.CONTINUOUS_HIGH_RES_MODE_2)

    def oneshot_low_res(self):
        self._set_mode(self.ONE_TIME_LOW_RES_MODE)

    def oneshot_high_res(self):
        self._set_mode(self.ONE_TIME_HIGH_RES_MODE_1)

    def oneshot_high_res2(self):
        self._set_mode(self.ONE_TIME_HIGH_RES_MODE_2)

    def set_sensitivity(self, sensitivity=69):
        """ Set the sensor sensitivity.
            Valid values are 31 (lowest) to 254 (highest), default is 69.
        """
        if sensitivity < 31:
            self.mtreg = 31
        elif sensitivity > 254:
            self.mtreg = 254
        else:
            self.mtreg = sensitivity
        self.power_on()
        self._set_mode(0x40 | (self.mtreg >> 5))
        self._set_mode(0x60 | (self.mtreg & 0x1f))
        self.power_down()

    def get_result(self):
        """ Return current measurement result in lx. """   
        data = self.bus.read_word_data(self.addr, self.mode)
        count = data >> 8 | (data&0xff)<<8
        mode2coeff =  2 if (self.mode & 0x03) == 0x01 else 1
        ratio = 1/(1.2 * (self.mtreg/69.0) * mode2coeff)
        return ratio*count

    def wait_for_result(self, additional=0):
        basetime = 0.018 if (self.mode & 0x03) == 0x03 else 0.128
        time.sleep(basetime * (self.mtreg/69.0) + additional)

    def do_measurement(self, mode, additional_delay=0):
        """ 
        Perform complete measurement using command
        specified by parameter mode with additional
        delay specified in parameter additional_delay.
        Return output value in Lx.
        """
        self.reset()
        self._set_mode(mode)
        self.wait_for_result(additional=additional_delay)
        return self.get_result()

    def measure_low_res(self, additional_delay=0):
        now=datetime.now()
        sample_ts=now.strftime("%H:%M:%S")
        return self.do_measurement(self.ONE_TIME_LOW_RES_MODE, additional_delay), sample_ts

    def measure_high_res(self, additional_delay=0):
        now=datetime.now()
        sample_ts=now.strftime("%H:%M:%S")
        return self.do_measurement(self.ONE_TIME_HIGH_RES_MODE_1, additional_delay), sample_ts

    def measure_high_res2(self, additional_delay=0):
        now=datetime.now()
        sample_ts=now.strftime("%H:%M:%S")
        return self.do_measurement(self.ONE_TIME_HIGH_RES_MODE_2, additional_delay), sample_ts



