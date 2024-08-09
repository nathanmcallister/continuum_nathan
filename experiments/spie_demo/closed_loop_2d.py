from continuum_arduino import ContinuumArduino
from continuum_aurora import ContinuumAurora
import torch
import numpy as np
from pathlib import Path
from scipy.spatial import distance
import time

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")
T_electrode_2_tip = np.loadtxt(Path("../../tools/T_electrode_2_tip"), delimiter=",", dtype=np.float64)

arduino = ContinuumArduino()
aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)
probe_list = ["0A"]
gain = 0.1
compression = 3
tolerance = 0.25
set_compression = 2

dls = -set_compression* np.ones(4, dtype=float)
arduino.write_dls(dls)
time.sleep(2)

range = 30

setpoints = np.array([[0, 5], [0, 10], [0, 15], [0, 20], [0, 25], [0, 30], [0, 0], 
                      [5, 0], [10, 0], [15, 0], [20, 0], [25, 0], [30, 0], [0, 0],
                      [0, -5], [0, -10], [0, -15], [0, -20], [0, -25], [0, -30], [0, 0],
                      [-5, 0], [-10, 0], [-15, 0], [-20, 0], [-25, 0], [-30, 0], [0, 0]]) # x,y

# setpoints = np.array([[1.58, 26.98], [0, 0]])

def get_error(setpoint):
    raw_trans = aurora.get_aurora_transforms(probe_list)
    T_tip_2_model = aurora.get_T_tip_2_model(raw_trans["0A"])
    T_electrode_2_model = T_tip_2_model @ T_electrode_2_tip
    elec_pos = np.array([T_electrode_2_model[0,3], T_electrode_2_model[1,3]])
    print(f"tip position = {np.array([T_tip_2_model[0,3], T_tip_2_model[1,3]])}")
    print(f"electrode position = {elec_pos}")
    error = (setpoint - elec_pos)
    return error

def to_setpoint(setpoint, is_compressed):
    global dls
    if is_compressed == 'c':
        dls = dls - 5
    elif is_compressed == 'e':
        dls = dls + 5
    else: pass
    arduino.write_dls(dls)
    error = get_error(setpoint)
    while distance.euclidean([0,0], error) > tolerance:
        # print(f"error = {error}\n")
        ex = error[0]; ey = error[1]
        dls[0] = dls[0] + (-gain * ex); dls[2] = dls[2] + (gain * ex)
        dls[1] = dls[1] + (-gain * ey); dls[3] = dls[3] + (gain * ey)
        arduino.write_dls(dls)
        error = get_error(setpoint)
    else:
        print("error is sufficiently low")

for setpoint in setpoints:
    to_setpoint(setpoint, 'c')
    print("reached setpoint compressed")
    to_setpoint(setpoint, 'e')
    print("reached setpoint extended")
    print("\nPress any key to continue...")
    input()

