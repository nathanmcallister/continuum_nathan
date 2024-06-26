#!/bin/python3
import numpy as np
import scipy.optimize as opt
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
import ANN
import camarillo_cc
import kinematics
import utils_cc
import utils_data

trajectory = np.loadtxt("output/trajectory.dat", delimiter=",")
trajectory_tensor = torch.tensor(trajectory)
num_points = trajectory.shape[1]
closed_loop_steps = 11
step_size = 0.5

camarillo_stiffness = np.loadtxt("../../tools/camarillo_stiffness", delimiter=",")
ka, kb, kt = camarillo_stiffness[0], camarillo_stiffness[1], camarillo_stiffness[2]
cable_positions_one_seg = [
    ((4, 0), (0, 4), (-4, 0), (0, -4)),
]
segment_stiffness_vals_one_seg = [(ka, kb)]
cable_stiffness_vals_one_seg = [(kt, kt, kt, kt)]
segment_lengths_one_seg = [64]

cable_positions_two_seg = cable_positions_one_seg + cable_positions_one_seg
segment_stiffness_vals_two_seg = (
    segment_stiffness_vals_one_seg + segment_stiffness_vals_one_seg
)
cable_stiffness_vals_two_seg = (
    cable_stiffness_vals_one_seg + cable_stiffness_vals_one_seg
)
segment_lengths_two_seg = segment_lengths_one_seg + segment_lengths_one_seg

one_seg_model = ANN.Model(
    input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
)
one_seg_model.load(
    "../model_learning/models/run_05_09_2024a/2024_05_10_01_59_02_noisy_one_seg.pt"
)
one_seg_model.model.eval()

two_seg_model = ANN.Model(
    input_dim=8, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
)
two_seg_model.load(
    "../model_learning/models/run_05_04_2024/2024_05_04_19_33_09_noisy_two_seg.pt"
)
two_seg_model.model.eval()

open_loop_dls = np.zeros((4, num_points))
model_pos = np.zeros((3, num_points))


def loss_fcn(
    dl: np.ndarray,
    model: ANN.Model,
    y_star: torch.tensor,
    dl_0: torch.tensor,
    weighting: Tuple[float, float],
) -> Tuple[float, np.ndarray]:

    dl_tensor = torch.tensor(dl, requires_grad=True)  # Convert numpy to tensor
    model.zero_grad()
    y_hat = model(dl_tensor)[0:3]

    e = y_hat - y_star  # Calculate position error
    delta_dl = dl_tensor - dl_0  # Calculate change in cable length

    loss = torch.dot(e, e) * weighting[0] + torch.dot(delta_dl, delta_dl) * weighting[1]

    loss.backward()  # Get gradient of loss function
    grad = dl_tensor.grad.numpy()

    return (loss.item(), grad)


for i in range(num_points):
    dl_0 = 0  # Set u_0
    if i == 0:
        dl_0 = torch.tensor(np.array([0.0] * 4))
    else:
        dl_0 = torch.tensor(open_loop_dls[:, i - 1].flatten())

    x_star = trajectory_tensor[:, i].flatten()

    optim_func = lambda x: loss_fcn(x, one_seg_model, x_star, dl_0, (5.0, 1.0))

    result = opt.minimize(optim_func, dl_0.numpy(), method="BFGS", jac=True)
    open_loop_dls[:, i] = result["x"]

open_loop_pos = np.zeros(trajectory.shape)
for i in range(num_points):
    model_pos[:, i] = (
        one_seg_model(torch.tensor(open_loop_dls[:, i]))[0:3].detach().numpy()
    )
    open_loop_dl_list = [tuple(open_loop_dls[:, i].tolist())]

    camarillo_params = camarillo_cc.no_slack_model(
        open_loop_dl_list,
        cable_positions_one_seg,
        segment_stiffness_vals_one_seg,
        cable_stiffness_vals_one_seg,
        segment_lengths_one_seg,
    )

    webster_params = utils_cc.camarillo_2_webster_params(
        camarillo_params, segment_lengths_one_seg
    )

    T_list = utils_cc.calculate_transforms(webster_params)
    open_loop_pos[:, i] = T_list[-1][0:3, 3]

model_error = model_pos - trajectory
model_error_norm = np.linalg.norm(model_error, axis=0)
model_rmse = np.sqrt((model_error_norm**2).mean())

open_loop_error = open_loop_pos - trajectory
open_loop_error_norm = np.linalg.norm(open_loop_error, axis=0)
open_loop_rmse = np.sqrt((open_loop_error_norm**2).mean())

closed_loop_pos = np.zeros((3, num_points * closed_loop_steps))


def calc_jacobian(u0, model):
    u0_tensor = torch.tensor(u0.flatten(), requires_grad=True)
    model.model.zero_grad()
    model_out = model(u0_tensor)
    unit_vectors = [
        torch.tensor(x)
        for x in [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ]
    ]

    jac = torch.stack(
        [
            torch.autograd.grad(model(u0_tensor), u0_tensor, vec)[0]
            for vec in unit_vectors
        ]
    ).numpy()

    return jac


for i in range(num_points):
    x0 = open_loop_pos[:, i]
    u0 = open_loop_dls[:, i]
    x_star = trajectory[:, i]

    jac = calc_jacobian(u0, one_seg_model)
    jac_pinv = np.linalg.pinv(jac)
    closed_loop_pos_inner = np.zeros((3, closed_loop_steps))
    delta = np.zeros_like(u0)
    for j in range(closed_loop_steps):

        open_loop_dl_list = [tuple((u0 + delta).tolist())]

        camarillo_params = camarillo_cc.no_slack_model(
            open_loop_dl_list,
            cable_positions_one_seg,
            segment_stiffness_vals_one_seg,
            cable_stiffness_vals_one_seg,
            segment_lengths_one_seg,
        )

        webster_params = utils_cc.camarillo_2_webster_params(
            camarillo_params, segment_lengths_one_seg
        )

        T_list = utils_cc.calculate_transforms(webster_params)
        closed_loop_pos_inner[:, j] = T_list[-1][0:3, 3]

        e = closed_loop_pos_inner[:, j] - x_star
        delta -= jac_pinv @ e * step_size

    closed_loop_pos[:, closed_loop_steps * i : closed_loop_steps * (i + 1)] = (
        closed_loop_pos_inner
    )

print(closed_loop_pos[0, 53 * closed_loop_steps : 54 * closed_loop_steps])

closed_loop_error = (
    closed_loop_pos[:, closed_loop_steps - 1 :: closed_loop_steps] - trajectory
)

closed_loop_error_norm = np.linalg.norm(closed_loop_error, axis=0)
closed_loop_rmse = np.sqrt((closed_loop_error_norm**2).mean())

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

ax = plt.figure().add_subplot(projection="3d")
ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], "-o", label="Trajectory")
ax.plot(
    model_pos[0, :],
    model_pos[1, :],
    model_pos[2, :],
    "-x",
    label=f"Model (RMSE: {model_rmse:.3f} mm)",
)
ax.plot(
    open_loop_pos[0, :],
    open_loop_pos[1, :],
    open_loop_pos[2, :],
    "-+",
    label=f"Open Loop (RMSE: {open_loop_rmse:.3f} mm)",
)
ax.plot(
    closed_loop_pos[0, closed_loop_steps - 1 :: closed_loop_steps],
    closed_loop_pos[1, closed_loop_steps - 1 :: closed_loop_steps],
    closed_loop_pos[2, closed_loop_steps - 1 :: closed_loop_steps],
    "-2",
    label=f"Closed Loop (RMSE: {closed_loop_rmse:.3f} mm)",
    color=colors[-1],
)
plt.xlim((-32, 32))
plt.ylim((-32, 32))
ax.set_zlim((0, 64))
plt.title("Simulation Closed Loop Trajectory Tracking")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
ax.set_zlabel("z (mm)")
plt.legend()

plt.figure()
plt.plot(trajectory[0, :], trajectory[1, :], "-o", label="Trajectory")
plt.plot(
    model_pos[0, :],
    model_pos[1, :],
    "-x",
    label=f"Model (RMSE: {model_rmse:.3f} mm)",
)
plt.plot(
    open_loop_pos[0, :],
    open_loop_pos[1, :],
    "-+",
    label=f"Open Loop (RMSE: {open_loop_rmse:.3f} mm)",
)
plt.plot(
    closed_loop_pos[0, closed_loop_steps - 1 :: closed_loop_steps],
    closed_loop_pos[1, closed_loop_steps - 1 :: closed_loop_steps],
    "-2",
    label=f"Closed Loop (RMSE: {closed_loop_rmse:.3f} mm)",
    color=colors[-1],
)

plt.title("Simulation Closed Loop Trajectory Tracking")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.legend(loc="lower right")

plt.figure()
plt.plot(
    closed_loop_pos[0, 80 * closed_loop_steps : (80 + 1) * closed_loop_steps]
    - trajectory[0, 80],
    label="$e_x$",
)
plt.plot(
    closed_loop_pos[1, 80 * closed_loop_steps : (80 + 1) * closed_loop_steps]
    - trajectory[1, 80],
    label="$e_y$",
)
plt.plot(
    closed_loop_pos[2, 80 * closed_loop_steps : (80 + 1) * closed_loop_steps]
    - trajectory[2, 80],
    label="$e_z$",
)
plt.plot(
    np.linalg.norm(
        closed_loop_pos[:, 80 * closed_loop_steps : (80 + 1) * closed_loop_steps]
        - trajectory[:, 80].reshape((-1, 1)),
        axis=0,
    ),
    label="$|e|$",
)
plt.legend()
plt.title("Simulation Closed Loop Error vs Step")
plt.xlabel("Step")
plt.ylabel("Position Error (mm)")

plt.figure()
plt.plot(model_error_norm, label="Model", color=colors[1])
plt.plot(open_loop_error_norm, label="Open Loop", color=colors[2])
plt.plot(closed_loop_error_norm, color=colors[-1], label="Closed Loop")
plt.title("Simulation Position Error Along Trajectory")
plt.xlabel("Step")
plt.ylabel("Error (mm)")
plt.legend(loc="upper left")
plt.show()
