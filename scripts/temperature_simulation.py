import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import cv2
import time

Pr = 1


def get_cost_processing(used_time):
    cost = 10 * used_time * Pr
    print(f"    >> [COST] Processing cost = {cost*1000} mW")
    return cost


def mean_absolute_error(gt_array, es_array):
    gt_average = np.mean(gt_array)
    es_average = np.mean(es_array)
    mae = np.abs(gt_average - es_average)
    return mae


def plot_temperature_array(input_array):
    average_temp = np.ones(len(input_array)) * np.mean(input_array)
    plt.plot(input_array, "b", label="Temperatures")
    plt.plot(average_temp, "r", label="Temp. Average")
    plt.title(f"Temperature sampled during {len(input_array)} seconds")
    plt.legend()
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temperature (degrees Celsius)")
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', linestyle='--')
    # plt.yticks(np.arange(17, 25, 0.1))

    plt.show()


def regulation_temp(input_temp):
    if input_temp < 18:
        input_temp = 18
    if input_temp > 24:
        input_temp = 24
    return input_temp


def main():
    print("Simulation")

    cv2.namedWindow("Temperature", cv2.WINDOW_NORMAL)
    temperature = 22
    temperatures_array = []
    max_samples = 60
    number_samples = 0
    used_time = 0
    while number_samples < max_samples:
        variance = np.random.uniform(-1, 1)
        temperature += variance
        temperature = regulation_temp(temperature)

        ####################
        #     PROCESS      #
        ####################
        t0 = time.time()
        temperatures_array.append(temperature)
        used_time += time.time() - t0

        image = np.zeros((500, 500, 3), np.uint8)
        cv2.putText(image, f"{temperature:.2f} C", (160, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow("Temperature", image)
        k = cv2.waitKey(1000)
        # k = cv2.waitKey(0)
        if k == 27:
            break
        number_samples += 1

    cost_processing = get_cost_processing(used_time)
    plot_temperature_array(temperatures_array)


if __name__ == '__main__':
    main()
