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
    "../model_learning/models/run_05_04_2024/2024_05_04_19_33_09_noisy_one_seg.pt"
)
one_seg_model.model.eval()

two_seg_model = ANN.Model(
    input_dim=8, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
)
two_seg_model.load(
    "../model_learning/models/run_05_04_2024/2024_05_04_19_33_09_noisy_two_seg.pt"
)
two_seg_model.model.eval()

boop = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)


def loss_fcn(
    dl: np.ndarray,
    model: ANN.Model,
    x_star: torch.tensor,
    dl_0: torch.tensor,
    weighting: Tuple[float, float],
) -> Tuple[float, np.ndarray]:

    dl_tensor = torch.tensor(dl, requires_grad=True)
    model.zero_grad()
    x_hat = model(dl_tensor)[0:3]

    e = x_hat - x_star
    delta_dl = dl_tensor - dl_0

    loss = torch.dot(e, e) * weighting[0] + torch.dot(delta_dl, delta_dl) * weighting[1]

    loss.backward()
    grad = dl_tensor.grad.numpy()

    return (loss.item(), grad)


dls = np.zeros((4, num_points))

for i in range(num_points):
    dl_0 = 0
    if i == 0:
        dl_0 = torch.tensor(np.array([0.0] * 4))
    else:
        dl_0 = torch.tensor(dls[:, i - 1].flatten())

    x_star = trajectory_tensor[:, i].flatten()

    optim_func = lambda x: loss_fcn(x, one_seg_model, x_star, dl_0, (5.0, 1.0))

    result = opt.minimize(optim_func, dl_0.numpy(), method="BFGS", jac=True)
    dls[:, i] = result["x"]

control_pos = np.zeros(trajectory.shape)
for i in range(num_points):
    point_dls = [tuple(dls[:, i].tolist())]

    camarillo_params = camarillo_cc.no_slack_model(
        point_dls,
        cable_positions_one_seg,
        segment_stiffness_vals_one_seg,
        cable_stiffness_vals_one_seg,
        segment_lengths_one_seg,
    )

    webster_params = utils_cc.camarillo_2_webster_params(
        camarillo_params, segment_lengths_one_seg
    )

    T_list = utils_cc.calculate_transforms(webster_params)
    control_pos[:, i] = T_list[-1][0:3, 3]

error = control_pos - trajectory
error_norm = np.linalg.norm(error, axis=0)
rmse = np.sqrt((error_norm**2).mean())

ax = plt.figure().add_subplot(projection="3d")
ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], "-o")
ax.plot(control_pos[0, :], trajectory[1, :], control_pos[2, :], "-o")
plt.xlim((-32, 32))
plt.ylim((-32, 32))
ax.set_zlim((0, 64))
plt.title(f"Learned Model Trajectory Tracking (RMSE: {rmse:.3f})")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
ax.set_zlabel("z (mm)")
plt.legend(["Reference Trajectory", "Tip Trajectory"])

plt.figure()
plt.plot(error_norm)
plt.title("Error Along Trajectory")
plt.xlabel("Trajectory Point")
plt.ylabel("Error (mm)")
plt.show()
