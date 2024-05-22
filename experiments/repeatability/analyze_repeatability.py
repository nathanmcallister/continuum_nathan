#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import time
import utils_data
import kinematics

container = utils_data.DataContainer()
container.file_import("output/data_2024_05_21_16_13_47.dat")
all_cable_deltas, all_pos, all_tang = container.to_numpy()


rounds = [(32 * x, 32 * (x + 1)) for x in range(17)]
avg_pos = []
max_error = []
rmse = []

plt.figure()

for i, round in enumerate(rounds):
    pos = all_pos[:, round[0] : round[1]]
    repeat_pos = pos[:, 1::2]
    repeat_pos = pos[:, np.abs(pos[0, :]) < 100]
    print(repeat_pos.shape)
    avg_pos.append(repeat_pos.mean(axis=1))
    plt.plot(repeat_pos[0, :], repeat_pos[1, :], "o", label=f"Group {i+1}")
    plt.text(avg_pos[-1][0], avg_pos[-1][1], f"{i}")

    error = repeat_pos - avg_pos[-1].reshape((-1, 1))
    error_mag = np.linalg.norm(error, axis=0)
    max_error.append(error_mag.max())
    rmse.append(np.sqrt((error_mag**2).mean()))

    print(
        f"{i}: ({avg_pos[-1][0]:.3f}, {avg_pos[-1][1]:.3f}, {avg_pos[-1][2]:.3f}), {max_error[-1]:.3f} mm, {rmse[-1]:.3f} mm"
    )
plt.title("Tip Position Clusters")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.show()
