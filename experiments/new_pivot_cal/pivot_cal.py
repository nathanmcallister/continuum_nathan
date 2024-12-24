#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import kinematics


def map_random(value, rand_range):
    return value * (rand_range[1] - rand_range[0]) + rand_range[0]


rng = np.random.default_rng()

t_tip_2_aurora_range = np.array([-500.0, 500.0])
t_coil_2_tip_std = 0

z_rotation_range = np.array([-np.pi, np.pi])
x_rotation_range = np.array([0, np.pi / 2])

t_tip_2_aurora = map_random(rng.random(3), t_tip_2_aurora_range)

t_coil_2_tip = np.array([0, 0, 25.0]) + t_coil_2_tip_std * rng.standard_normal(3)

q_tip_2_aurora = rng.standard_normal((4, 1))
q_tip_2_aurora = q_tip_2_aurora / np.linalg.norm(q_tip_2_aurora)
R_tip_2_aurora = kinematics.quat_2_dcm(q_tip_2_aurora)

q_coil_2_tip = rng.standard_normal((4, 1))
q_coil_2_tip = q_coil_2_tip / np.linalg.norm(q_coil_2_tip)
R_coil_2_tip = kinematics.quat_2_dcm(q_coil_2_tip)

T_tip_2_aurora = np.identity(4)
T_tip_2_aurora[:3, :3] = R_tip_2_aurora
T_tip_2_aurora[:3, 3] = t_tip_2_aurora

T_coil_2_tip = np.identity(4)
T_coil_2_tip[:3, :3] = R_coil_2_tip
T_coil_2_tip[:3, 3] = t_coil_2_tip

num_meas = 500
coil_pos = np.zeros((3, num_meas))
coil_quat = np.zeros((4, num_meas))

for i in range(num_meas):

    z_rotations = map_random(rng.random(2), z_rotation_range)
    x_rotation = map_random(rng.random(), x_rotation_range)

    T_rotation = np.identity(4)
    T_rotation[:3, :3] = (
        kinematics.rotz(z_rotations[1])
        @ kinematics.rotx(x_rotation)
        @ kinematics.rotz(z_rotations[0])
    )

    T_coil_2_aurora = T_tip_2_aurora @ T_rotation @ T_coil_2_tip

    coil_pos[:, i] = T_coil_2_aurora[:3, 3]
    coil_quat[:, i] = kinematics.dcm_2_quat(T_coil_2_aurora[:3, :3])


# ax = plt.figure().add_subplot(projection="3d")
# ax.plot(coil_pos[0, :], coil_pos[1, :], coil_pos[2, :], "o")
# plt.show()
