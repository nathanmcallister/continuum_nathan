#!/bin/python3
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import kinematics
from utils_data import parse_aurora_csv
from typing import List

axis_file = "../../data/tip_cals/tip_cal_06_26_24c.csv"
pivot_file = "../../data/tip_cals/tip_cal_06_26_24b.csv"

axis_transforms = parse_aurora_csv(axis_file)
positions = np.nan * np.zeros((3, len(axis_transforms["0A"])))
for idx, transform in enumerate(axis_transforms["0A"]):
    positions[:, idx] = transform[1]

centroid = np.mean(positions, axis=1)
cov = (positions - centroid.reshape((3, 1))) @ np.transpose(
    positions - centroid.reshape((3, 1))
)
vals, vecs = np.linalg.eig(cov)

sorted_vals = sorted(enumerate(vals.tolist()), key=lambda x: x[1], reverse=True)
normal_vectors = [vecs[:, val[0]] for val in sorted_vals]


def optim_function(
    x: np.ndarray, points: np.ndarray, normal_vectors: List[np.ndarray]
) -> float:
    # x[0]: r, x[1:4]: (xc, yc, zc)
    r = x[0]
    xc = x[1:4]

    centered_points = points - xc.reshape((3, 1))
    num_points = centered_points.shape[1]
    projected_points = np.zeros((3, num_points))
    for i in range(num_points):
        a_proj = np.dot(normal_vectors[0], centered_points[:, i])
        b_proj = np.dot(normal_vectors[1], centered_points[:, i])
        projected_points[:, i] = a_proj * normal_vectors[0] + b_proj * normal_vectors[1]

    dist = np.linalg.norm(projected_points, axis=0)
    error = dist - r

    return (error**2).mean()


x0 = np.concatenate([np.array([1], dtype=float), centroid])
print(
    opt.minimize(
        lambda x: optim_function(x, positions, normal_vectors), x0, method="BFGS"
    )
)


ax = plt.figure().add_subplot(projection="3d")
ax.plot(positions[0, :], positions[1, :], positions[2, :], "o")
for idx, vector in enumerate(normal_vectors):
    ax.quiver(
        centroid[0],
        centroid[1],
        centroid[2],
        vector[0],
        vector[1],
        vector[2],
        color=f"C{idx+1}",
    )
ax.set_box_aspect(
    (np.ptp(positions[0, :]), np.ptp(positions[1, :]), np.ptp(positions[2, :]))
)
plt.show()

pivot_transforms = parse_aurora_csv(pivot_file)
