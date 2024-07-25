#!/bin/python3
import numpy as np
import torch
from scipy.optimize import minimize
from pathlib import Path
from multi_input import MultiInputModel

trajectory_file = "./trajectories/trajectory.dat"
trajectory_positions = np.loadtxt(Path(trajectory_file), delimiter=",")

model = MultiInputModel(4, 1, 1, [32, 32])
model.load("../multi_input_learning/models/07_25_2024/2024_07_25_11_02_40.pt")

previous_input = np.zeros(4)

def optim_fcn(x: np.ndarray, model: MultiInputModel, point: np.ndarray, previous_input: np.ndarray):

