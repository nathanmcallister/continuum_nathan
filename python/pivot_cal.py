#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from kinematics import quat_2_dcm
from utils_data import parse_aurora_csv

# Parameters
pivot_filename = "../data/tip_cals/tip_cal_07_02_24a.csv"
std_threshold = 3
show_plots = True
output_filename = "../tools/penprobe_07_02_24a"


# Local functions
def solve_system(
    coil_quat: np.ndarray,
    coil_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function creates/ solves the overdefined system of eqns for penprobe + pivot

    Args:
        coil_quat: 4 x N array containing coil quaternions
        coil_pos: 3 x N array containing coil positions

    Returns:
        The penprobe and pivot vectors
    """
    num_meas = coil_quat.shape[1]
    assert num_meas == coil_pos.shape[1]

    # Form overdefined system of equations
    A = np.zeros((3 * num_meas, 6))
    b = np.zeros((3 * num_meas))
    for i in range(num_meas):
        A[3 * i : 3 * (i + 1), :] = np.concatenate(
            [quat_2_dcm(coil_quat[:, i]), -np.identity(3)], axis=1
        )
        b[3 * i : 3 * (i + 1)] = -coil_pos[:, i]

    # Solve system
    out = np.linalg.lstsq(A, b, rcond=None)

    # Extract and return
    penprobe = out[0][:3]
    pivot = out[0][3:]
    return penprobe, pivot


def get_error(
    coil_quat: np.ndarray,
    coil_pos: np.ndarray,
    penprobe: np.ndarray,
    pivot: np.ndarray,
) -> np.ndarray:
    """
    Finds error between coil + rotated penprobe vs pivot point.

    Args:
        coil_quat: 4 x N array containing coil quaternions
        coil_pos: 4 x N array containing coil positions
        penprobe: Coil to tip vector
        pivot: Position in Aurora space pen pivots about

    Returns:
        An array containing the position error for each coil measurement
    """
    # Setup
    num_meas = coil_quat.shape[1]
    assert num_meas == coil_pos.shape[1]
    error = np.zeros(num_meas)

    # Go through all points and find deviation from pivot
    for i in range(num_meas):
        tip_pos = coil_pos[:, i] + quat_2_dcm(coil_quat[:, i]) @ penprobe
        error[i] = np.linalg.norm(tip_pos - pivot)

    return error


# Load data
aurora_data = parse_aurora_csv(pivot_filename)["0A"]
num_meas = len(aurora_data)
coil_quat = np.zeros((4, num_meas))
coil_pos = np.zeros((3, num_meas))

for i in range(num_meas):
    coil_quat[:, i] = aurora_data[i][0]
    coil_pos[:, i] = aurora_data[i][1]


# Iterate to remove outliers (greater than 3 standard deviations above mean error)
previous_length = np.inf
current_length = coil_quat.shape[1]
rmse = np.inf
iteration = 1

while previous_length != current_length:
    previous_length = current_length

    penprobe, pivot = solve_system(coil_quat, coil_pos)

    # Perform error calculations
    error = get_error(coil_quat, coil_pos, penprobe, pivot)
    mean_error = error.mean()
    error_std = np.std(error)
    rmse = np.sqrt(np.mean(error**2))

    print(f"Iteration {iteration} RMSE: {rmse.item():.3f}")

    # Find points that are within threshold
    good_measurements = error - mean_error < std_threshold * error_std

    # Update coil_quat and pos
    coil_quat = coil_quat[:, good_measurements]
    coil_pos = coil_pos[:, good_measurements]

    current_length = coil_quat.shape[1]
    iteration += 1

# Save file
np.savetxt(output_filename, penprobe, delimiter=",")

# Plotting
if show_plots:
    num_meas = coil_quat.shape[1]
    full_data = np.concatenate([coil_pos, pivot.reshape((3, 1))], axis=1)
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(coil_pos[0, :], coil_pos[1, :], coil_pos[2, :], label="Coil Positions")
    ax.set_box_aspect(
        (np.ptp(full_data[0, :]), np.ptp(full_data[1, :]), np.ptp(full_data[2, :]))
    )
    for i in range(0, num_meas, 2):
        vec = quat_2_dcm(coil_quat[:, i]) @ penprobe
        legend_label = "Penprobe Vectors" if i == 0 else "_Penprobe Vectors"
        ax.quiver(*coil_pos[:, i], *vec, label=legend_label, color="C1")

    ax.scatter(pivot[0], pivot[1], pivot[2], label="Pivot Point", color="C2")
    plt.title(f"Pivot Calibration (RMSE: {rmse.item():.3f} mm)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    plt.legend()
    plt.figure()
    plt.plot(error)
    plt.title("Error of Data Points")
    plt.xlabel("Point")
    plt.ylabel("Error (mm)")
    plt.show()
