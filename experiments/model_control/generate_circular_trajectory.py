#!/bin/python3
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

t = np.arange(0, 2 * np.pi, np.pi / 16)
r = (
    np.concatenate(
        [
            np.ones(len(t)),
            2 * np.ones(len(t)),
            3 * np.ones(len(t)),
            4 * np.ones(len(t)),
            5 * np.ones(len(t)),
            6 * np.ones(len(t)),
        ]
    )
    * 5
)
t = np.concatenate([t] * 6)

pos = np.zeros((3, len(t)))
pos[0, :] = r * np.cos(t)
pos[1, :] = r * np.sin(t)
l = 60

for i in range(len(t)):
    x = pos[0, i]
    y = pos[1, i]
    if np.abs(x) <= 0.001 and np.abs(y) <= 0.001:
        pos[2, i] = l
    elif np.abs(x) >= 0.0001:
        phi = np.arctan2(y, x)

        optim_func = lambda theta: np.sqrt(
            (l / theta * (1 - np.cos(theta)) * np.cos(phi) - x) ** 2
        )

        result = opt.minimize(optim_func, 0.7)
        theta = result["x"]
        pos[2, i] = (l / theta * np.sin(theta)).item()
    else:
        phi = np.arctan2(y, x)

        optim_func = lambda theta: np.sqrt(
            (l / theta * (1 - np.cos(theta)) * np.sin(phi) - y) ** 2
        )

        result = opt.minimize(optim_func, 0.01)
        theta = result["x"]
        pos[2, i] = (l / theta * np.sin(theta)).item()

np.savetxt("output/circular_trajectory.dat", pos, delimiter=",")

ax = plt.figure().add_subplot(projection="3d")
ax.plot(pos[0, :], pos[1, :], pos[2, :], "-o")
plt.xlim((-l / 2, l / 2))
plt.ylim((-l / 2, l / 2))
ax.set_zlim((0, l))
plt.title("Desired Tip Trajectory")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
ax.set_zlabel("z (mm)")

plt.figure()
plt.plot(pos[0, :], pos[1, :], "-o")
plt.title("Desired Tip Trajectory")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.xlim((-32, 32))
plt.ylim((-32, 32))
plt.show()
