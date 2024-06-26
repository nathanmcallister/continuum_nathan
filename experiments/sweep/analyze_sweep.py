#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import utils_data

container = utils_data.DataContainer()
container.file_import("output/data_2024_05_21_14_57_12.dat")
cable_deltas, pos, tang = container.to_numpy()

phi_desired = np.arctan2(cable_deltas[3, :], cable_deltas[2, :])
phi_meas = np.arctan2(pos[1, :], pos[0, :])

good_idx = [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

num_loops = int(len(phi_meas) / 32)
error = np.zeros((num_loops, 32))
for i in range(num_loops):
    error[i, :] = np.unwrap(phi_meas[32 * i : 32 * (i + 1)]) - np.unwrap(
        phi_desired[32 * i : 32 * (i + 1)]
    )

avg_error = error[good_idx, :].mean(axis=0)
avg_std = np.std(error[good_idx, :], axis=0, ddof=1)

plt.figure()
plt.plot(180 / np.pi * phi_desired)
plt.plot(180 / np.pi * phi_meas)
plt.title("Measured and Desired Angle by Measurement")
plt.xlabel("Measurement")
plt.ylabel("Angle (degrees)")
plt.legend(["Desired Angle", "Measured Angle"], loc="upper right")

plt.figure()
plt.plot(
    180 / np.pi * np.unwrap(phi_desired[:32]),
    180 / np.pi * avg_error,
    label="Mean Angular Error",
)
plt.fill_between(
    180 / np.pi * np.unwrap(phi_desired[:32]),
    180 / np.pi * (avg_error - avg_std),
    180 / np.pi * (avg_error + avg_std),
    alpha=0.3,
    label="_Angular Error Std",
)
plt.plot(
    [0, 0, np.nan, 90, 90, np.nan, 180, 180, np.nan, 270, 270, np.nan, 360, 360],
    [-45, 45, np.nan, -45, 45, np.nan, -45, 45, np.nan, -45, 45, np.nan, -45, 45],
    label="Cable Axes",
)
plt.legend(loc="upper right")
plt.title("Angular Error vs Desired Angle")
plt.xlabel("Desired Angle (degrees)")
plt.ylabel("Angular Error (degrees)")
plt.show()
