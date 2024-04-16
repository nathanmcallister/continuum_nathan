#!/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import kinematics
import continuum_aurora

AURORA_CIRCLE_FILE = "../../data/base_circle_04_16_24a.csv"
AURORA_TOP_FILE = "../../data/base_top_04_16_24a.csv"

T_AURORA_2_MODEL_FILE = "../../tools/T_aurora_2_model"
T_TIP_2_COIL_FILE = "../../tools/T_tip_2_coil"

PENPROBE_FILE = "../../tools/penprobe7"

T_aurora_2_model = np.loadtxt(T_AURORA_2_MODEL_FILE, delimiter=",")
T_tip_2_coil = np.loadtxt(T_TIP_2_COIL_FILE, delimiter=",")

penprobe = np.loadtxt(PENPROBE_FILE, delimiter=",")

def base_circle() -> np.ndarray:
    x = np.zeros((2,1))

    circle_df = pd.read_table(AURORA_CIRCLE_FILE, sep=',', header=None)
    num_probes = len(pd.unique(circle_df[2]))

    penprobe_index = "0A"
    pen_table = circle_df[:][circle_df[2] == penprobe_index]

    pen_quat = pen_table.iloc[:, 3:7].to_numpy().transpose()
    pen_coil_pos = pen_table.iloc[:, 7:10].to_numpy().transpose()

    num_meas = pen_coil_pos.shape[1]


base_circle()
