#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import utils_data

data_file = "output/dynamic_2024_04_23_15_15_46.dat"

container = utils_data.DataContainer()
container.file_import(data_file)

dls = np.concatenate([x.reshape((-1, 1)) for x in container.inputs], axis=1)
outputs = np.concatenate([x.reshape((-1, 1)) for x in container.outputs], axis=1)

pos = outputs[0:3, :]
tang = outputs[3:, :]


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pos[0,:], pos[1, :], pos[2, :])
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

xmean = (xlim[1] + xlim[0]) / 2
ymean = (ylim[1] + ylim[0]) / 2
zmean = (zlim[1] + zlim[0]) / 2

xrange = xlim[1] - xlim[0]
yrange = ylim[1] - ylim[0]
zrange = zlim[1] - zlim[0]

bound_size = max(xrange, yrange, zrange)

ax.set_xlim((xmean - bound_size, xmean + bound_size))
ax.set_ylim((ymean - bound_size, ymean + bound_size))
ax.set_zlim((0, zmean + bound_size))

plt.show()
