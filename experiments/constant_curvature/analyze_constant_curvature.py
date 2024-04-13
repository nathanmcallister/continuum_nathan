#!/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils_cc

meas_data = np.loadtxt("meas.csv", delimiter=",")
zero_data = np.loadtxt("zero.csv", delimiter=",")

print(meas_data[433:435:,9])
print(meas_data[433:435,10])

print(np.argwhere(np.abs(meas_data[:, 9]) > 100))
print(np.argwhere(np.abs(meas_data[:, 10]) > 100))

phi_meas = np.arctan2(meas_data[:, 10] - zero_data[0:5, 3].mean(), meas_data[:, 9] - zero_data[0:5, 2].mean())
print(phi_meas[0:10])

desired_pos = np.zeros((meas_data.shape[0], 3))

for i in range(meas_data.shape[0]):
    T_list = utils_cc.calculate_transform([(64, meas_data[i, 4], meas_data[i, 3])])
    T = T_list[0]
    desired_pos[i, :] = T[0:3, 3]

plt.figure(1)
plt.scatter(zero_data[:, 2], zero_data[:, 3])
plt.show()

plt.figure(2)
plt.scatter(desired_pos[:, 0], desired_pos[:, 1])
plt.scatter(meas_data[np.logical_and(np.abs(meas_data[:, 9]) < 100, np.abs(meas_data[:, 10]) < 100), 9], meas_data[np.logical_and(np.abs(meas_data[:, 9]) < 100, np.abs(meas_data[:, 10]) < 100), 10])
plt.show()

plt.figure(3)
plt.scatter(meas_data[:, 9], meas_data[:, 10], meas_data[:, 11])
plt.show()
