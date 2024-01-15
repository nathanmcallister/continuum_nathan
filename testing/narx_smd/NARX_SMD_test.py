from torch.utils.data import DataLoader
import NARX
from spring_mass_damper import SMD, dlsim
import numpy as np
import matplotlib.pyplot as plt
import random

# Definitions
# Simulation properties
TRAIN_DATAPOINTS = 70000
TEST_DATAPOINTS = 10000

# NARX properties
NUM_PREVIOUS_OBSERVATIONS = 1
NUM_PREVIOUS_ACTIONS = 1

# SMD properties
K = 1  # Spring stiffness
B = 0.707  # Damping
M = 1  # Mass
DT = 0.01  # Timestep

OBSERVATION_TYPE = "full_state"

# Setup
if OBSERVATION_TYPE == "full_state":
    obs_dim = 2
else:
    obs_dim = 1

# Define models
spring_mass_damper = SMD(K, M, B, DT, OBSERVATION_TYPE)
narx_model = NARX.Model(
    obs_dim,
    1,
    [16],
    num_previous_observations=NUM_PREVIOUS_OBSERVATIONS,
    num_previous_actions=NUM_PREVIOUS_ACTIONS,
    lr=1e-3,
)

# Create training and testing inputs
u_train = np.random.randn(1, TRAIN_DATAPOINTS) * 2
u_test = np.random.randn(1, TEST_DATAPOINTS) * 2

# Zero out random inputs (not necessary, but adds some spice)
for i in range(TRAIN_DATAPOINTS):
    if random.random() <= 0.5:
        u_train[:, i] = np.zeros((u_train.shape[0], 1), dtype=float)
for i in range(TEST_DATAPOINTS):
    if random.random() <= 0.5:
        u_test[:, i] = np.zeros((u_train.shape[0], 1), dtype=float)

# Initial states for dlsim
x0_train = np.random.randn(2, 1)
x0_test = np.random.randn(2, 1)

# Simulate SMD system
y_train = dlsim(
    spring_mass_damper.A,
    spring_mass_damper.B,
    spring_mass_damper.C,
    spring_mass_damper.D,
    u_train,
    x0_train,
)
y_test = dlsim(
    spring_mass_damper.A,
    spring_mass_damper.B,
    spring_mass_damper.C,
    spring_mass_damper.D,
    u_test,
    x0_test,
)

# Create Datasets and DataLoaders
train_dataset = NARX.Dataset(
    y_train,
    u_train,
    num_previous_observations=NUM_PREVIOUS_OBSERVATIONS,
    num_previous_actions=NUM_PREVIOUS_ACTIONS,
)
test_dataset = NARX.Dataset(
    y_test,
    u_test,
    num_previous_observations=NUM_PREVIOUS_OBSERVATIONS,
    num_previous_actions=NUM_PREVIOUS_ACTIONS,
)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Training
narx_model.train(
    train_dataloader,
    test_dataloader,
    num_epochs=10,
    model_save_path="narx_model.pt"
)

# Comparison/ evaluation
u_comparison = np.random.randn(1, 2048) * 2
y_comparison = dlsim(
    spring_mass_damper.A,
    spring_mass_damper.B,
    spring_mass_damper.C,
    spring_mass_damper.D,
    u_comparison,
    np.zeros((2, 1), dtype=float),
)

# Perform comparison
y_narx_comparison = narx_model.comparison_test(
    u_comparison, y_comparison, "narx_model.pt"
)

# Plotting
plt.plot(list(range(y_comparison.shape[1])),
         y_comparison.transpose(), label="Truth")
plt.plot(
    list(range(y_comparison.shape[1])),
    y_narx_comparison.transpose(),
    label="NARX"
)
plt.title("NARX Predicted State and True State vs Time")
plt.xlabel("Time Step (k)")
plt.ylabel("System State (m and m/s)")
plt.legend()
plt.show()
