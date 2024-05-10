#!/bin/python3
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import mike_cc
import utils_cc
import utils_data

show_figures = True
save_figures = False
optim = True

# Data filenames
phi_zero_file = "output/sweep_phi/zero_2024_04_23_09_09_33.dat"
theta_zero_file = "output/sweep_theta/zero_2024_04_23_08_44_51.dat"

phi_meas_file = "output/sweep_phi/meas_2024_04_23_09_09_33.dat"
theta_meas_file = "output/sweep_theta/meas_2024_04_23_08_44_51.dat"

# Load data into containers
phi_zero_container = utils_data.DataContainer()
phi_zero_container.file_import(phi_zero_file)

theta_zero_container = utils_data.DataContainer()
theta_zero_container.file_import(theta_zero_file)

phi_meas_container = utils_data.DataContainer()
phi_meas_container.file_import(phi_meas_file)

theta_meas_container = utils_data.DataContainer()
theta_meas_container.file_import(theta_meas_file)

# Extract zero pose from zero containers
phi_zero_pos = np.nan * np.zeros((3, phi_zero_container.num_measurements))
phi_zero_tang = np.nan * np.zeros((3, phi_zero_container.num_measurements))
theta_zero_pos = np.nan * np.zeros((3, theta_zero_container.num_measurements))
theta_zero_tang = np.nan * np.zeros((3, theta_zero_container.num_measurements))

for i in range(phi_zero_container.num_measurements):
    phi_zero_pos[:, i] = phi_zero_container.outputs[i][0:3]
    phi_zero_tang[:, i] = phi_zero_container.outputs[i][3:]

for i in range(theta_zero_container.num_measurements):
    theta_zero_pos[:, i] = theta_zero_container.outputs[i][0:3]
    theta_zero_tang[:, i] = theta_zero_container.outputs[i][3:]

# Extract meas pose from meas containers
phi_meas_pos = np.nan * np.zeros((3, phi_meas_container.num_measurements))
phi_meas_tang = np.nan * np.zeros((3, phi_meas_container.num_measurements))
phi_meas_dls = np.nan * np.zeros(
    (phi_meas_container.num_cables, phi_meas_container.num_measurements)
)
theta_meas_pos = np.nan * np.zeros((3, theta_meas_container.num_measurements))
theta_meas_tang = np.nan * np.zeros((3, theta_meas_container.num_measurements))
theta_meas_dls = np.nan * np.zeros(
    (theta_meas_container.num_cables, theta_meas_container.num_measurements)
)

for i in range(phi_meas_container.num_measurements):
    phi_meas_pos[:, i] = phi_meas_container.outputs[i][0:3]
    phi_meas_tang[:, i] = phi_meas_container.outputs[i][3:]
    phi_meas_dls[:, i] = phi_meas_container.inputs[i]

for i in range(theta_meas_container.num_measurements):
    theta_meas_pos[:, i] = theta_meas_container.outputs[i][0:3]
    theta_meas_tang[:, i] = theta_meas_container.outputs[i][3:]
    theta_meas_dls[:, i] = theta_meas_container.inputs[i]

# Find faulty measurements
faulty_phi_zero = np.argwhere((np.linalg.norm(phi_zero_pos, axis=0) > 100))
phi_zero_pos = np.ma.array(phi_zero_pos, mask=False)
phi_zero_tang = np.ma.array(phi_zero_tang, mask=False)
phi_zero_pos.mask[:, faulty_phi_zero] = True
phi_zero_tang.mask[:, faulty_phi_zero] = True

faulty_theta_zero = np.argwhere((np.linalg.norm(theta_zero_pos, axis=0) > 100))
theta_zero_pos = np.ma.array(theta_zero_pos, mask=False)
theta_zero_tang = np.ma.array(theta_zero_tang, mask=False)
theta_zero_pos.mask[:, faulty_theta_zero] = True
theta_zero_tang.mask[:, faulty_theta_zero] = True

faulty_phi_meas = np.argwhere((np.linalg.norm(phi_meas_pos, axis=0) > 100))
phi_meas_pos = np.ma.array(phi_meas_pos, mask=False)
phi_meas_tang = np.ma.array(phi_meas_tang, mask=False)
phi_meas_dls = np.ma.array(phi_meas_dls, mask=False)
phi_meas_pos.mask[:, faulty_phi_meas] = True
phi_meas_tang.mask[:, faulty_phi_meas] = True
phi_meas_dls.mask[:, faulty_phi_meas] = True

faulty_theta_meas = np.argwhere((np.linalg.norm(theta_meas_pos, axis=0) > 100))
theta_meas_pos = np.ma.array(theta_meas_pos, mask=False)
theta_meas_tang = np.ma.array(theta_meas_tang, mask=False)
theta_meas_dls = np.ma.array(theta_meas_dls, mask=False)
theta_meas_pos.mask[:, faulty_theta_meas] = True
theta_meas_tang.mask[:, faulty_theta_meas] = True
theta_meas_dls.mask[:, faulty_theta_meas] = True

# Extract zero data stats
avg_phi_zero_pos = phi_zero_pos.mean(axis=1)
avg_theta_zero_pos = theta_zero_pos.mean(axis=1)

std_phi_zero_pos = np.std(phi_zero_pos, axis=1)
std_theta_zero_pos = np.std(theta_zero_pos, axis=1)

# Create combined dataset for model fitting
combined_pos = np.ma.compress_cols(
    np.ma.concatenate((phi_meas_pos, theta_meas_pos), axis=1)
)
combined_tang = np.ma.compress_cols(
    np.ma.concatenate((phi_meas_tang, theta_meas_tang), axis=1)
)
combined_dls = np.ma.compress_cols(
    np.ma.concatenate((phi_meas_dls, theta_meas_dls), axis=1)
)

# Use CC model to fit xc, yc, l, dl0, dl1 (because mike only uses two cables, can only fit for two cables at a time)
xc0, yc0, l0, dl0, dl1 = 0, 0, 64, 0, 0
opt_x0 = np.array([xc0, yc0, l0, dl0, dl1])
cable_positions = [(4, 0), (0, 4), (-4, 0), (0, -4)]


def pos_optim_func(x, cables=[0, 1], return_positions=False):
    model_pos = np.ma.array(np.nan * np.zeros(combined_pos.shape), mask=False)

    for i in range(combined_pos.shape[1]):
        if not np.ma.is_masked(combined_pos[:, i]):
            webster_params, _ = mike_cc.one_seg_forward_kinematics(
                combined_dls[cables[0], i] + x[3],
                combined_dls[cables[1], i] + x[4],
                cable_positions[cables[0]],
                cable_positions[cables[1]],
                x[2],
            )

            T = utils_cc.calculate_transform(webster_params)
            model_pos[0, i] = T[0, 3] + x[0]
            model_pos[1, i] = T[1, 3] + x[1]
            model_pos[2, i] = T[2, 3]
        else:
            print(i)
            model_pos.mask[:, i] = True

    if not return_positions:
        return np.sqrt((np.linalg.norm(combined_pos - model_pos, axis=0) ** 2).mean())

    return model_pos


if optim:
    opt_function = lambda x: pos_optim_func(x, [0, 1], False)
    res = opt.minimize(opt_function, opt_x0, method="BFGS")

    xf = res["x"]
    rmse = res["fun"]
    optim_str = "Fitted"

else:
    optim_str = "Unfitted"
    xf = opt_x0
    rmse = pos_optim_func(xf)

model_pos = pos_optim_func(xf, return_positions=True)

# Generate contour plot of error
theta_steps = 12
theta = np.linspace(0, np.pi / 2, theta_steps + 1)
theta = theta[1:]

phi_steps = 24
phi = np.linspace(0, 2 * np.pi, phi_steps + 1)
phi = phi[:-1]

phi_avg_error = np.nan * np.zeros((theta_steps * phi_steps))
theta_avg_error = np.nan * np.zeros((theta_steps * phi_steps))

for i in range(theta_steps * phi_steps):
    phi_dl = phi_meas_dls[:, 5 * i : 5 * (i + 1)].mean(axis=1)
    phi_webster_params, _ = mike_cc.one_seg_forward_kinematics(
        phi_dl[0] + xf[3],
        phi_dl[1] + xf[4],
        cable_positions[0],
        cable_positions[1],
        xf[2],
    )
    T_phi = utils_cc.calculate_transform(phi_webster_params)
    phi_model_pos = T_phi[0:3, 3] + np.array([xf[0], xf[1], 0])

    phi_avg_error[i] = np.sqrt(
        ((phi_meas_pos[:, 5 * i : 5 * (i + 1)] - phi_model_pos.reshape((-1, 1))) ** 2)
        .sum(axis=0)
        .mean()
    )

    theta_dl = theta_meas_dls[:, 5 * i : 5 * (i + 1)].mean(axis=1)
    theta_webster_params, _ = mike_cc.one_seg_forward_kinematics(
        theta_dl[0] + xf[3],
        theta_dl[1] + xf[4],
        cable_positions[0],
        cable_positions[1],
        xf[2],
    )
    T_theta = utils_cc.calculate_transform(theta_webster_params)
    theta_model_pos = T_theta[0:3, 3] + np.array([xf[0], xf[1], 0])

    theta_avg_error[i] = np.sqrt(
        (
            (theta_meas_pos[:, 5 * i : 5 * (i + 1)] - theta_model_pos.reshape((-1, 1)))
            ** 2
        )
        .sum(axis=0)
        .mean()
    )

plt.figure()
plt.contourf(
    180 / np.pi * phi,
    180 / np.pi * theta,
    0.5
    * (
        phi_avg_error.reshape((theta_steps, phi_steps))
        + theta_avg_error.reshape((phi_steps, theta_steps)).transpose()
    ),
)
plt.colorbar()
plt.title("RMS Contour of Constant Curvature Model")
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")


# Plot combined data
plt.figure()
plt.scatter(model_pos[0, :], model_pos[1, :])
plt.scatter(combined_pos[0, :], combined_pos[1, :])
plt.scatter(xf[0], xf[1])
plt.title(f"{optim_str} X-Z Plane Projection of Model and Measured Position")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.legend(["Model", "Measured", "Robot Base"])
if save_figures:
    plt.savefig(f"figures/{optim_str.lower()}_xy.png")

plt.figure()
plt.plot(np.linalg.norm(combined_pos - model_pos, axis=0), "o")
plt.title(f"{optim_str} Model Error Norm by Measurement (rmse: {rmse:.3f})")
plt.xlabel("Measurement")
plt.ylabel("Error (mm)")
if save_figures:
    plt.savefig(f"figures/{optim_str.lower()}_error_norm.png")

plt.figure()
plt.subplot(311)
plt.plot((combined_pos[0, :] - model_pos[0, :]), "o")
plt.title(f"{optim_str} Error of Constant Curvature Model by Axis")
plt.ylabel("X Error (mm)")
plt.subplot(312)
plt.plot((combined_pos[1, :] - model_pos[1, :]), "o")
plt.ylabel("Y Error (mm)")
plt.subplot(313)
plt.plot((combined_pos[2, :] - model_pos[2, :]), "o")
plt.xlabel("Measurement")
plt.ylabel("Z Error (mm)")
if save_figures:
    plt.savefig(f"figures/{optim_str.lower()}_error_axis.png")

# Plot zero data
plt.figure()
plt.hist(phi_zero_pos[0, :], bins=50, alpha=0.7)
plt.hist(theta_zero_pos[0, :], bins=50, alpha=0.7)
plt.title("Histogram of X Position of Zero Cable Lengths")
plt.xlabel("X (mm)")
plt.ylabel("Count")
plt.legend(
    [
        f"Phi (mean: {avg_phi_zero_pos[0]:.3f} mm, std: {std_phi_zero_pos[0]:.3f} mm)",
        f"Theta (mean: {avg_theta_zero_pos[0]:.3f} mm, std: {std_theta_zero_pos[0]:.3f} mm)",
    ]
)
if save_figures:
    plt.savefig("figures/zero_x.png")

plt.figure()
plt.hist(phi_zero_pos[1, :], bins=50, alpha=0.7)
plt.hist(theta_zero_pos[1, :], bins=50, alpha=0.7)
plt.title("Histogram of Y Position of Zero Cable Lengths")
plt.xlabel("Y (mm)")
plt.ylabel("Count")
plt.legend(
    [
        f"Phi (mean: {avg_phi_zero_pos[1]:.3f} mm, std: {std_phi_zero_pos[1]:.3f} mm)",
        f"Theta (mean: {avg_theta_zero_pos[1]:.3f} mm, std: {std_theta_zero_pos[1]:.3f} mm)",
    ]
)
if save_figures:
    plt.savefig("figures/zero_y.png")

plt.figure()
plt.hist(phi_zero_pos[2, :], bins=50, alpha=0.7)
plt.hist(theta_zero_pos[2, :], bins=50, alpha=0.7)
plt.title("Histogram of Z Position of Zero Cable Lengths")
plt.xlabel("Z (mm)")
plt.ylabel("Count")
plt.legend(
    [
        f"Phi (mean: {avg_phi_zero_pos[2]:.3f} mm, std: {std_phi_zero_pos[2]:.3f} mm)",
        f"Theta (mean: {avg_theta_zero_pos[2]:.3f} mm, std: {std_theta_zero_pos[2]:.3f} mm)",
    ]
)
if save_figures:
    plt.savefig("figures/zero_z.png")

plt.figure()
plt.scatter(phi_zero_pos[0, :], phi_zero_pos[1, :], alpha=0.7)
plt.scatter(theta_zero_pos[0, :], theta_zero_pos[1, :], alpha=0.7)
plt.title("X-Y Position of Zero Cable Lengths")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.legend(["Phi", "Theta"])
if save_figures:
    plt.savefig("figures/zero_xy.png")

if show_figures:
    plt.show()
