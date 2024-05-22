#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import utils_data

num_iterations = 10

# File inputs
container = utils_data.DataContainer()
container.file_import("training_data/clean_1_seg_2024_05_09_15_55_05.dat")
container.clean()
_, pos, _ = container.to_numpy()
# Training loss vs epoch
clean_train_loss_one_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_09_2024a/clean_train_loss_one_seg_{i}.dat", delimiter=","
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
noisy_train_loss_one_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_09_2024a/noisy_train_loss_one_seg_{i}.dat", delimiter=","
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
clean_train_loss_two_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_09_2024a/clean_train_loss_two_seg_{i}.dat", delimiter=","
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
noisy_train_loss_two_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_09_2024a/noisy_train_loss_two_seg_{i}.dat", delimiter=","
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)

# Validation loss vs epoch
clean_validation_loss_one_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_09_2024a/clean_validation_loss_one_seg_{i}.dat",
            delimiter=",",
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
noisy_validation_loss_one_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_09_2024a/noisy_validation_loss_one_seg_{i}.dat",
            delimiter=",",
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
clean_validation_loss_two_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_09_2024a/clean_validation_loss_two_seg_{i}.dat",
            delimiter=",",
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
noisy_validation_loss_two_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_09_2024a/noisy_validation_loss_two_seg_{i}.dat",
            delimiter=",",
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)

# Test loss vs model
clean_clean_pos_loss_one_seg = np.loadtxt(
    "output/run_05_09_2024a/c_c_1_pos_test_loss.dat", delimiter=","
)
clean_clean_tang_loss_one_seg = np.loadtxt(
    "output/run_05_09_2024a/c_c_1_tang_test_loss.dat", delimiter=","
)
noisy_clean_pos_loss_one_seg = np.loadtxt(
    "output/run_05_09_2024a/n_c_1_pos_test_loss.dat", delimiter=","
)
noisy_clean_tang_loss_one_seg = np.loadtxt(
    "output/run_05_09_2024a/n_c_1_tang_test_loss.dat", delimiter=","
)
clean_clean_pos_loss_two_seg = np.loadtxt(
    "output/run_05_09_2024a/c_c_2_pos_test_loss.dat", delimiter=","
)
clean_clean_tang_loss_two_seg = np.loadtxt(
    "output/run_05_09_2024a/c_c_2_tang_test_loss.dat", delimiter=","
)
noisy_clean_pos_loss_two_seg = np.loadtxt(
    "output/run_05_09_2024a/n_c_2_pos_test_loss.dat", delimiter=","
)
noisy_clean_tang_loss_two_seg = np.loadtxt(
    "output/run_05_09_2024a/n_c_2_tang_test_loss.dat", delimiter=","
)

combined_pos_loss = np.concatenate(
    [
        clean_clean_pos_loss_one_seg.reshape((-1, 1)),
        noisy_clean_pos_loss_one_seg.reshape((-1, 1)),
        clean_clean_pos_loss_two_seg.reshape((-1, 1)),
        noisy_clean_pos_loss_two_seg.reshape((-1, 1)),
    ],
    axis=1,
)

avg_pos_loss = combined_pos_loss.mean(axis=0)

combined_tang_loss = np.concatenate(
    [
        clean_clean_tang_loss_one_seg.reshape((-1, 1)),
        noisy_clean_tang_loss_one_seg.reshape((-1, 1)),
        clean_clean_tang_loss_two_seg.reshape((-1, 1)),
        noisy_clean_tang_loss_two_seg.reshape((-1, 1)),
    ],
    axis=1,
)

avg_tang_loss = combined_tang_loss.mean(axis=0)

# Means and stdevs
avg_clean_train_loss_one_seg = clean_train_loss_one_seg.mean(axis=0)
std_clean_train_loss_one_seg = np.std(clean_train_loss_one_seg, axis=0, ddof=1)

avg_noisy_train_loss_one_seg = noisy_train_loss_one_seg.mean(axis=0)
std_noisy_train_loss_one_seg = np.std(noisy_train_loss_one_seg, axis=0, ddof=1)

avg_clean_train_loss_two_seg = clean_train_loss_two_seg.mean(axis=0)
std_clean_train_loss_two_seg = np.std(clean_train_loss_two_seg, axis=0, ddof=1)

avg_noisy_train_loss_two_seg = noisy_train_loss_two_seg.mean(axis=0)
std_noisy_train_loss_two_seg = np.std(noisy_train_loss_two_seg, axis=0, ddof=1)

avg_clean_validation_loss_one_seg = clean_validation_loss_one_seg.mean(axis=0)
std_clean_validation_loss_one_seg = np.std(
    clean_validation_loss_one_seg, axis=0, ddof=1
)

avg_noisy_validation_loss_one_seg = noisy_validation_loss_one_seg.mean(axis=0)
std_noisy_validation_loss_one_seg = np.std(
    noisy_validation_loss_one_seg, axis=0, ddof=1
)

avg_clean_validation_loss_two_seg = clean_validation_loss_two_seg.mean(axis=0)
std_clean_validation_loss_two_seg = np.std(
    clean_validation_loss_two_seg, axis=0, ddof=1
)

avg_noisy_validation_loss_two_seg = noisy_validation_loss_two_seg.mean(axis=0)
std_noisy_validation_loss_two_seg = np.std(
    noisy_validation_loss_two_seg, axis=0, ddof=1
)


# Plotting
plt.figure(1)
x = np.arange(1, len(avg_clean_train_loss_one_seg) + 1)
plt.plot(x, avg_clean_train_loss_one_seg, label="Mean Clean One Segment Loss")
plt.fill_between(
    x,
    avg_clean_train_loss_one_seg - std_clean_train_loss_one_seg,
    avg_clean_train_loss_one_seg + std_clean_train_loss_one_seg,
    alpha=0.3,
    label="_clean one segment std",
)
plt.plot(x, avg_noisy_train_loss_one_seg, label="Mean Noisy One Segment Loss")
plt.fill_between(
    x,
    avg_noisy_train_loss_one_seg - std_noisy_train_loss_one_seg,
    avg_noisy_train_loss_one_seg + std_noisy_train_loss_one_seg,
    alpha=0.3,
    label="_noisy one segment std",
)
plt.plot(x, avg_clean_train_loss_two_seg, label="Mean Clean Two Segment Loss")
plt.fill_between(
    x,
    avg_clean_train_loss_two_seg - std_clean_train_loss_two_seg,
    avg_clean_train_loss_two_seg + std_clean_train_loss_two_seg,
    alpha=0.3,
    label="_clean two segment std",
)
plt.plot(x, avg_noisy_train_loss_two_seg, label="Mean Noisy Two Segment Loss")
plt.fill_between(
    x,
    avg_noisy_train_loss_two_seg - std_noisy_train_loss_two_seg,
    avg_noisy_train_loss_two_seg + std_noisy_train_loss_two_seg,
    alpha=0.3,
    label="_noisy two segment std",
)
plt.title("Average Batch Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")

plt.figure(2)
plt.semilogy(x, avg_clean_train_loss_one_seg, label="Mean Clean One Segment Loss")
plt.fill_between(
    x,
    avg_clean_train_loss_one_seg - std_clean_train_loss_one_seg,
    avg_clean_train_loss_one_seg + std_clean_train_loss_one_seg,
    alpha=0.3,
    label="_clean one segment std",
)
plt.semilogy(x, avg_noisy_train_loss_one_seg, label="Mean Noisy One Segment Loss")
plt.fill_between(
    x,
    avg_noisy_train_loss_one_seg - std_noisy_train_loss_one_seg,
    avg_noisy_train_loss_one_seg + std_noisy_train_loss_one_seg,
    alpha=0.3,
    label="_noisy one segment std",
)
plt.semilogy(x, avg_clean_train_loss_two_seg, label="Mean Clean Two Segment Loss")
plt.fill_between(
    x,
    avg_clean_train_loss_two_seg - std_clean_train_loss_two_seg,
    avg_clean_train_loss_two_seg + std_clean_train_loss_two_seg,
    alpha=0.3,
    label="_clean two segment std",
)
plt.semilogy(x, avg_noisy_train_loss_two_seg, label="Mean Noisy Two Segment Loss")
plt.fill_between(
    x,
    avg_noisy_train_loss_two_seg - std_noisy_train_loss_two_seg,
    avg_noisy_train_loss_two_seg + std_noisy_train_loss_two_seg,
    alpha=0.3,
    label="_noisy two segment std",
)
plt.title("Average Batch Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")

plt.figure(3)
plt.plot(x, avg_clean_validation_loss_one_seg, label="Mean Clean One Segment Loss")
plt.fill_between(
    x,
    avg_clean_validation_loss_one_seg - std_clean_validation_loss_one_seg,
    avg_clean_validation_loss_one_seg + std_clean_validation_loss_one_seg,
    alpha=0.3,
    label="_clean one segment std",
)
plt.plot(x, avg_noisy_validation_loss_one_seg, label="Mean Noisy One Segment Loss")
plt.fill_between(
    x,
    avg_noisy_validation_loss_one_seg - std_noisy_validation_loss_one_seg,
    avg_noisy_validation_loss_one_seg + std_noisy_validation_loss_one_seg,
    alpha=0.3,
    label="_noisy one segment std",
)
plt.plot(x, avg_clean_validation_loss_two_seg, label="Mean Clean Two Segment Loss")
plt.fill_between(
    x,
    avg_clean_validation_loss_two_seg - std_clean_validation_loss_two_seg,
    avg_clean_validation_loss_two_seg + std_clean_validation_loss_two_seg,
    alpha=0.3,
    label="_clean two segment std",
)
plt.plot(x, avg_noisy_validation_loss_two_seg, label="Mean Noisy Two Segment Loss")
plt.fill_between(
    x,
    avg_noisy_validation_loss_two_seg - std_noisy_validation_loss_two_seg,
    avg_noisy_validation_loss_two_seg + std_noisy_validation_loss_two_seg,
    alpha=0.3,
    label="_noisy two segment std",
)
plt.title("Average Batch Validation Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")

plt.figure(4)
plt.semilogy(x, avg_clean_validation_loss_one_seg, label="Mean Clean One Segment Loss")
plt.fill_between(
    x,
    avg_clean_validation_loss_one_seg - std_clean_validation_loss_one_seg,
    avg_clean_validation_loss_one_seg + std_clean_validation_loss_one_seg,
    alpha=0.3,
    label="_clean one segment std",
)
plt.semilogy(x, avg_noisy_validation_loss_one_seg, label="Mean Noisy One Segment Loss")
plt.fill_between(
    x,
    avg_noisy_validation_loss_one_seg - std_noisy_validation_loss_one_seg,
    avg_noisy_validation_loss_one_seg + std_noisy_validation_loss_one_seg,
    alpha=0.3,
    label="_noisy one segment std",
)
plt.semilogy(x, avg_clean_validation_loss_two_seg, label="Mean Clean Two Segment Loss")
plt.fill_between(
    x,
    avg_clean_validation_loss_two_seg - std_clean_validation_loss_two_seg,
    avg_clean_validation_loss_two_seg + std_clean_validation_loss_two_seg,
    alpha=0.3,
    label="_clean two segment std",
)
plt.semilogy(x, avg_noisy_validation_loss_two_seg, label="Mean Noisy Two Segment Loss")
plt.fill_between(
    x,
    avg_noisy_validation_loss_two_seg - std_noisy_validation_loss_two_seg,
    avg_noisy_validation_loss_two_seg + std_noisy_validation_loss_two_seg,
    alpha=0.3,
    label="_noisy two segment std",
)
plt.title("Average Batch Validation Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")

plt.figure(5)
for i in range(0, 10, 2):
    plt.hist(clean_clean_pos_loss_one_seg[i, :], 100, alpha=0.3, label=f"Model {i+1}")
plt.title("Position Error on One Segment Clean Data Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(6)
for i in range(0, 10, 2):
    plt.hist(
        180 / np.pi * np.sqrt(clean_clean_tang_loss_one_seg[i, :]),
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Orientation Error on One Segment Clean Data Model")
plt.xlabel("Norm of Error (degrees)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(7)
for i in range(0, 10, 2):
    plt.hist(noisy_clean_pos_loss_one_seg[i, :], 100, alpha=0.3, label=f"Model {i+1}")
plt.title("Position Error on One Segment Noisy Data Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(8)
for i in range(0, 10, 2):
    plt.hist(
        180 / np.pi * np.sqrt(noisy_clean_tang_loss_one_seg[i, :]),
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Orientation Error on One Segment Noisy Data Model")
plt.xlabel("Norm of Error (degrees)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(9)
for i in range(0, 10, 2):
    plt.hist(clean_clean_pos_loss_two_seg[i, :], 100, alpha=0.3, label=f"Model {i+1}")
plt.title("Position Error on Two Segment Clean Data Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(10)
for i in range(0, 10, 2):
    plt.hist(
        180 / np.pi * np.sqrt(clean_clean_tang_loss_two_seg[i, :]),
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Orientation Error on Two Segment Clean Data Model")
plt.xlabel("Norm of Error (degrees)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(11)
for i in range(0, 10, 2):
    plt.hist(noisy_clean_pos_loss_two_seg[i, :], 100, alpha=0.3, label=f"Model {i+1}")
plt.title("Position Error on Two Segment Noisy Data Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(12)
for i in range(0, 10, 2):
    plt.hist(
        180 / np.pi * np.sqrt(noisy_clean_tang_loss_two_seg[i, :]),
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Orientation Error on Two Segment Noisy Data Model")
plt.xlabel("Norm of Error (degrees)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(13)
plt.boxplot(
    combined_pos_loss[:, :2],
    showfliers=False,
    labels=[
        f"Clean (Mean: {avg_pos_loss[0].item():.3f} mm)",
        f"Noisy (Mean: {avg_pos_loss[1].item():.3f} mm)",
    ],
)
plt.title("Position Error of One-Segment Models")
plt.xlabel("Dataset")
plt.ylabel("Position Error (mm)")

plt.figure(14)
plt.boxplot(
    180 / np.pi * combined_tang_loss[:, :2],
    showfliers=False,
    labels=[
        f"Clean (Mean: ${180 / np.pi * avg_tang_loss[0].item():.3f}^\circ$)",
        f"Noisy (Mean: ${180 / np.pi * avg_tang_loss[1].item():.3f}^\circ$)",
    ],
)
plt.title("Orientation Error of One-Segment Models")
plt.xlabel("Dataset")
plt.ylabel("Orientation Error (degrees)")

plt.figure(15)
plt.boxplot(
    combined_pos_loss[:, 2:],
    showfliers=False,
    labels=[
        f"Clean (Mean: {avg_pos_loss[2].item():.3f} mm)",
        f"Noisy (Mean: {avg_pos_loss[3].item():.3f} mm)",
    ],
)
plt.title("Position Error of Two-Segment Models")
plt.xlabel("Dataset")
plt.ylabel("Position Error (mm)")

plt.figure(16)
plt.boxplot(
    180 / np.pi * combined_tang_loss[:, 2:],
    showfliers=False,
    labels=[
        f"Clean (Mean: ${180 / np.pi * avg_tang_loss[2].item():.3f}^\circ$)",
        f"Noisy (Mean: ${180 / np.pi * avg_tang_loss[3].item():.3f}^\circ$)",
    ],
)
plt.title("Orientation Error of Two-Segment Models")
plt.xlabel("Dataset")
plt.ylabel("Orientation Error (degrees)")

plt.figure()
plt.plot(pos[0, :], pos[1, :], "o", alpha=0.3)
plt.title("Simulation Training Data Position Measurements")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.show()
