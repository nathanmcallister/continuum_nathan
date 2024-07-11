#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from utils_data import DataContainer

container = DataContainer()
container.file_import("output/data_2024_07_03_14_14_04.dat")

cable_dls, pos, tang = container.to_numpy()

for i in range(25):
    ax = plt.figure(i).add_subplot(projection="3d")
    ax.plot(
        pos[0, 160 * i : 160 * (i + 1)],
        pos[1, 160 * i : 160 * (i + 1)],
        pos[2, 160 * i : 160 * (i + 1)],
    )
    plt.xlim((-32, 32))
    plt.ylim((-32, 32))
    ax.set_zlim((0, 64))
    ax.set_box_aspect((1, 1, 1))
plt.show()
