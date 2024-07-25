#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt

cable_trajectory = np.loadtxt("output/nathan_cable_trajectory.dat", delimiter=",")

plt.figure()
plt.plot(cable_trajectory[0, :])
plt.plot(cable_trajectory[1, :])
plt.plot(cable_trajectory[2, :])
plt.plot(cable_trajectory[3, :])
plt.show()
