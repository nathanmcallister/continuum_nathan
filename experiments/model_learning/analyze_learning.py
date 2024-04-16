#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt

train_loss = np.loadtxt("train_loss.dat", delimiter=",")
test_loss = np.loadtxt("test_loss.dat", delimiter=",")

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()
