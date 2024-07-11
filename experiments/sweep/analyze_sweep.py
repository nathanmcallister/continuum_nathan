#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import utils_data

from pyplot_units import radians, degrees


multi_positions = np.loadtxt("output/multi_sweep.dat", delimiter=",")
center = multi_positions.mean(axis=1)
angular_steps = 128
positions = np.zeros((3, angular_steps))
for i in range(angular_steps):
    positions[:, i] = multi_positions[:, i::angular_steps].mean(axis=1)

phi = np.arange(angular_steps) * 2 * np.pi / angular_steps
measured_phi = np.unwrap(
    np.arctan2(positions[1, :] - center[1], positions[0, :] - center[0])
)

plt.figure()
plt.plot(multi_positions[0, :], multi_positions[1, :])
plt.plot(positions[0, :], positions[1, :])
plt.title("X-Y Projection of Measured Positions")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

fig, ax = plt.subplots(1)
ax.plot(phi * radians, phi * radians, xunits=degrees, yunits=degrees)
ax.plot(phi * radians, measured_phi * radians, xunits=degrees, yunits=degrees)
plt.title("Commanded and Measured Angle")
plt.xlabel("Commanded Angle (degrees)")
plt.ylabel("Angle (degrees)")
plt.legend(["Commanded Angle", "Measured Angle"])

plt.figure()
plt.plot(phi * radians, positions[2, :], xunits=degrees)
plt.title("Z Position vs Commanded Angle")
plt.xlabel("Commanded Angle (degrees)")
plt.ylabel("z (mm)")

ax = plt.figure().add_subplot(projection="polar")
ax.plot(phi, np.abs(measured_phi - phi) * radians, yunits=degrees)
plt.ylabel("")
plt.title("Angular Error Norm vs Commanded Angle")
plt.show()
