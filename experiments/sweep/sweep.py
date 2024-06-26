#!/bin/python3
import numpy as np
import time
import continuum_aurora
import continuum_arduino
import kinematics
import utils_data

max_displacement = 12
angular_steps = 32
radial_steps = 16
wait = 0.25
num_meas = angular_steps * radial_steps

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

phi = np.arange(0, angular_steps) * 2 * np.pi / angular_steps
dl = (np.arange(0, radial_steps) + 1) * max_displacement / radial_steps

phi_mesh, dl_mesh = np.meshgrid(phi, dl)
phi_sweep = phi_mesh.flatten()
dl_sweep = dl_mesh.flatten()

cable_deltas = np.concatenate(
    [
        (-dl_sweep * np.cos(phi_sweep)).reshape((1, -1)),
        (-dl_sweep * np.sin(phi_sweep)).reshape((1, -1)),
        (dl_sweep * np.cos(phi_sweep)).reshape((1, -1)),
        (dl_sweep * np.sin(phi_sweep)).reshape((1, -1)),
    ],
    axis=0,
)

arduino = continuum_arduino.init_arduino()
aurora = continuum_aurora.init_aurora()

pos = np.nan * np.zeros((3, num_meas))
tang = np.nan * np.zeros((3, num_meas))

motor_setpoints = continuum_arduino.load_motor_setpoints("../../tools/motor_setpoints")
continuum_arduino.write_motor_vals(arduino, motor_setpoints)
time.sleep(wait)
print("    dl |     phi")
for i in range(num_meas):
    print(f"{dl_sweep[i]:6.3f} | {180 / np.pi * phi_sweep[i]:7.3f}")
    motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(
        cable_deltas[:, i], motor_setpoints
    )
    continuum_arduino.write_motor_vals(arduino, motor_vals)
    time.sleep(wait)
    transforms = continuum_aurora.get_aurora_transforms(aurora, ["0A"])
    T = continuum_aurora.get_T_tip_2_model(
        transforms["0A"], T_aurora_2_model, T_tip_2_coil
    )

    pos[:, i] = T[0:3, 3]
    tang[:, i] = kinematics.dcm_2_tang(T[0:3, 0:3])

continuum_arduino.write_motor_vals(arduino, motor_setpoints)
container = utils_data.DataContainer()
container.prefix = "output/data"
container.set_date_and_time()
container.from_raw_data(
    container.date, container.time, 4, num_meas, cable_deltas, pos, tang
)
container.file_export()
