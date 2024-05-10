#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt

num_iterations = 10

# File inputs
# Training loss vs epoch
clean_train_loss_one_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_04_2024/clean_train_loss_one_seg_{i}.dat", delimiter=","
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
noisy_train_loss_one_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_04_2024/noisy_train_loss_one_seg_{i}.dat", delimiter=","
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
clean_train_loss_two_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_04_2024/clean_train_loss_two_seg_{i}.dat", delimiter=","
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
noisy_train_loss_two_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_04_2024/noisy_train_loss_two_seg_{i}.dat", delimiter=","
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)

# Validation loss vs epoch
clean_validation_loss_one_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_04_2024/clean_validation_loss_one_seg_{i}.dat",
            delimiter=",",
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
noisy_validation_loss_one_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_04_2024/noisy_validation_loss_one_seg_{i}.dat",
            delimiter=",",
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
clean_validation_loss_two_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_04_2024/clean_validation_loss_two_seg_{i}.dat",
            delimiter=",",
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)
noisy_validation_loss_two_seg = np.concatenate(
    [
        np.loadtxt(
            f"output/run_05_04_2024/noisy_validation_loss_two_seg_{i}.dat",
            delimiter=",",
        ).reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)

# Test loss vs model
clean_clean_pos_loss_one_seg = np.loadtxt(
    "output/run_05_04_2024/c_c_1_pos_test_loss.dat", delimiter=","
)
clean_clean_tang_loss_one_seg = np.loadtxt(
    "output/run_05_04_2024/c_c_1_tang_test_loss.dat", delimiter=","
)
noisy_clean_pos_loss_one_seg = np.loadtxt(
    "output/run_05_04_2024/n_c_1_pos_test_loss.dat", delimiter=","
)
noisy_clean_tang_loss_one_seg = np.loadtxt(
    "output/run_05_04_2024/n_c_1_tang_test_loss.dat", delimiter=","
)
clean_clean_pos_loss_two_seg = np.loadtxt(
    "output/run_05_04_2024/c_c_2_pos_test_loss.dat", delimiter=","
)
clean_clean_tang_loss_two_seg = np.loadtxt(
    "output/run_05_04_2024/c_c_2_tang_test_loss.dat", delimiter=","
)
noisy_clean_pos_loss_two_seg = np.loadtxt(
    "output/run_05_04_2024/n_c_2_pos_test_loss.dat", delimiter=","
)
noisy_clean_tang_loss_two_seg = np.loadtxt(
    "output/run_05_04_2024/n_c_2_tang_test_loss.dat", delimiter=","
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
combined_tang_loss = np.concatenate(
    [
        clean_clean_tang_loss_one_seg.reshape((-1, 1)),
        noisy_clean_tang_loss_one_seg.reshape((-1, 1)),
        clean_clean_tang_loss_two_seg.reshape((-1, 1)),
        noisy_clean_tang_loss_two_seg.reshape((-1, 1)),
    ],
    axis=1,
)

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
    plt.hist(
        clean_clean_pos_loss_one_seg[i, clean_clean_pos_loss_one_seg[i, :] < 2],
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Zoomed Position Error on One Segment Clean Data Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(10)
for i in range(0, 10, 2):
    plt.hist(
        180
        / np.pi
        * np.sqrt(
            clean_clean_tang_loss_one_seg[
                i, 180 / np.pi * np.sqrt(clean_clean_tang_loss_one_seg[i, :]) <= 10
            ]
        ),
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Zoomed Orientation Error on One Segment Clean Data Model")
plt.xlabel("Norm of Error (degrees)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(11)
for i in range(0, 10, 2):
    plt.hist(
        noisy_clean_pos_loss_one_seg[i, noisy_clean_pos_loss_one_seg[i, :] <= 2],
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Zoomed Position Error on One Segment Noisy Data Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(12)
for i in range(0, 10, 2):
    plt.hist(
        180
        / np.pi
        * np.sqrt(
            noisy_clean_tang_loss_one_seg[
                i, 180 / np.pi * np.sqrt(noisy_clean_tang_loss_one_seg[i, :]) <= 10
            ]
        ),
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Zoomed Orientation Error on One Segment Noisy Data Model")
plt.xlabel("Norm of Error (degrees)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(13)
for i in range(0, 10, 2):
    plt.hist(clean_clean_pos_loss_two_seg[i, :], 100, alpha=0.3, label=f"Model {i+1}")
plt.title("Position Error on Two Segment Clean Data Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(14)
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

plt.figure(15)
for i in range(0, 10, 2):
    plt.hist(noisy_clean_pos_loss_two_seg[i, :], 100, alpha=0.3, label=f"Model {i+1}")
plt.title("Position Error on Two Segment Noisy Data Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(16)
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

plt.figure(17)
for i in range(0, 10, 2):
    plt.hist(
        clean_clean_pos_loss_two_seg[i, clean_clean_pos_loss_two_seg[i, :] < 2],
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Zoomed Position Error on Two Segment Clean Data Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(18)
for i in range(0, 10, 2):
    plt.hist(
        180
        / np.pi
        * np.sqrt(
            clean_clean_tang_loss_two_seg[
                i, 180 / np.pi * np.sqrt(clean_clean_tang_loss_two_seg[i, :]) <= 10
            ]
        ),
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Zoomed Orientation Error on Two Segment Clean Data Model")
plt.xlabel("Norm of Error (degrees)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(19)
for i in range(0, 10, 2):
    plt.hist(
        noisy_clean_pos_loss_two_seg[i, noisy_clean_pos_loss_two_seg[i, :] <= 2],
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Zoomed Position Error on Two Segment Noisy Data Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure(20)
for i in range(0, 10, 2):
    plt.hist(
        180
        / np.pi
        * np.sqrt(
            noisy_clean_tang_loss_two_seg[
                i, 180 / np.pi * np.sqrt(noisy_clean_tang_loss_two_seg[i, :]) <= 10
            ]
        ),
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Zoomed Orientation Error on Two Segment Noisy Data Model")
plt.xlabel("Norm of Error (degrees)")
plt.ylabel("Count")
plt.legend(loc="upper right")
plt.show()
