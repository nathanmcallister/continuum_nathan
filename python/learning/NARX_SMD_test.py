import torch
from torch.utils.data import DataLoader
from NARX import *
from NARX_Dataset import *
from spring_mass_damper import *
import numpy as np
import matplotlib.pyplot as plt
import random

TRAIN_DATAPOINTS = 70000
TEST_DATAPOINTS = 10000

num_previous_observations = 1
num_previous_actions = 1

narx_model = NARX(2, 1, [16], num_previous_observations=num_previous_observations, num_previous_actions=num_previous_actions, lr=1e-3)
spring_mass_damper = SMD(1, 1, .707, 0.01, 'full_state')

u_train = np.random.randn(1, TRAIN_DATAPOINTS) * 2 
for i in range(TRAIN_DATAPOINTS):
    if random.random() <= .5:
        u_train[:,i] = np.zeros((u_train.shape[0],1), dtype=float)
u_test = np.random.randn(1, TEST_DATAPOINTS) * 2

for i in range(TEST_DATAPOINTS):
    if random.random() <= .5:
        u_test[:,i] = np.zeros((u_train.shape[0],1), dtype=float)

x0_train = np.random.randn(2,1)
x0_test = np.random.randn(2,1)

y_train = dlsim(spring_mass_damper.A, spring_mass_damper.B, spring_mass_damper.C, spring_mass_damper.D, u_train, x0_train)
y_test = dlsim(spring_mass_damper.A, spring_mass_damper.B, spring_mass_damper.C, spring_mass_damper.D, u_test, x0_test)

train_dataset = NARX_Dataset(y_train, u_train, num_previous_observations=num_previous_observations, num_previous_actions=num_previous_actions)
test_dataset = NARX_Dataset(y_test, u_test,  num_previous_observations=num_previous_observations, num_previous_actions=num_previous_actions)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

#narx_model.train(train_dataloader, test_dataloader, num_epochs=10, model_save_path="narx_model.pt")

u_comparison = np.random.randn(1,2048) * 2
y_comparison = dlsim(spring_mass_damper.A, spring_mass_damper.B, spring_mass_damper.C, spring_mass_damper.D, u_comparison, np.zeros((2,1),dtype=float))
#u_comparison = u_test
#y_comparison = y_test

y_narx_comparison = narx_model.comparison_test(u_comparison, y_comparison, "narx_model.pt")
#y_narx_comparison = narx_model.comparison_test(u_comparison, y_comparison)

plt.plot(list(range(y_comparison.shape[1])), y_comparison.transpose(), label="Truth")
plt.plot(list(range(y_comparison.shape[1])), y_narx_comparison.transpose(), label="NARX")
plt.title("NARX Predicted State and True State vs Time")
plt.xlabel("Time Step (k)")
plt.ylabel("System State (m and m/s)")
plt.legend()
plt.show()
