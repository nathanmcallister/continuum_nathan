#!/bin/python3
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from ANN import Model, Dataset, PoseLoss
from utils_data import DataContainer
from typing import List, Tuple


def train(
    dataset: Dataset, iterations: int = 5
) -> Tuple[List[Model], List[Tuple[float, float]]]:

    models = []
    losses = []
    power = int(np.log(len(dataset)) / np.log(2))
    for i in range(iterations):
        print(f"Power {power} | Model {i+1}")
        model = Model(
            8,
            6,
            [32, 32],
            loss=PoseLoss(),
            save_path=f"models/data_size/{power}_{i}.pt",
        )
        train_dataset, validation_dataset = random_split(dataset, [0.75, 0.25])

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        validation_dataloader = DataLoader(
            validation_dataset, batch_size=64, shuffle=True
        )

        loss = model.train(
            train_dataloader,
            validation_dataloader,
            checkpoints=True,
            save_model=True,
        )

        models.append(model)
        losses.append(loss)

    return models, losses


def test(
    cable_deltas: np.ndarray, pos: np.ndarray, tang: np.ndarray, models: List[Model]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    cable_deltas_tensor = torch.tensor(np.transpose(cable_deltas))

    test_losses = []
    for model in models:
        model.model.eval()
        with torch.no_grad():
            model_output = model(cable_deltas_tensor)

        model_pos = np.transpose(model_output[:, :3].numpy())
        model_tang = np.transpose(model_output[:, 3:].numpy())

        pos_error = np.linalg.norm(model_pos - pos, axis=0)
        tang_error = np.linalg.norm(model_tang - tang, axis=0)

        test_losses.append((pos_error, tang_error))

    return test_losses


if __name__ == "__main__":
    # Load data
    test_container = DataContainer()
    test_container.file_import("./test_data/9_2024_07_31_08_57_01.dat")
    test_deltas, test_pos, test_tang = test_container.to_numpy()

    dataset_13 = Dataset("./training_data/13_2024_07_31_08_55_18.dat")
    dataset_14 = Dataset("./training_data/14_2024_07_31_08_55_51.dat")
    dataset_15 = Dataset("./training_data/15_2024_07_31_08_56_59.dat")

    # Train models
    models_13, loss_13 = train(dataset_13)
    models_14, loss_14 = train(dataset_14)
    models_15, loss_15 = train(dataset_15)

    # Extract losses
    train_loss_13 = np.array([x[0] for x in loss_13])
    validation_loss_13 = np.array([x[1] for x in loss_13])

    train_loss_14 = np.array([x[0] for x in loss_14])
    validation_loss_14 = np.array([x[1] for x in loss_14])

    train_loss_15 = np.array([x[0] for x in loss_15])
    validation_loss_15 = np.array([x[1] for x in loss_15])

    # Output losses to files
    np.savetxt("output/data_size/train_loss_13.dat", train_loss_13, delimiter=",")
    np.savetxt(
        "output/data_size/validation_loss_13.dat", validation_loss_13, delimiter=","
    )

    np.savetxt("output/data_size/train_loss_14.dat", train_loss_14, delimiter=",")
    np.savetxt(
        "output/data_size/validation_loss_14.dat", validation_loss_14, delimiter=","
    )

    np.savetxt("output/data_size/train_loss_15.dat", train_loss_15, delimiter=",")
    np.savetxt(
        "output/data_size/validation_loss_15.dat", validation_loss_15, delimiter=","
    )

    # Test models
    test_loss_13 = test(test_deltas, test_pos, test_tang, models_13)
    test_loss_14 = test(test_deltas, test_pos, test_tang, models_14)
    test_loss_15 = test(test_deltas, test_pos, test_tang, models_15)

    # Extract losses
    pos_loss_13 = np.concatenate([x[0].reshape((1, -1)) for x in test_loss_13], axis=0)
    tang_loss_13 = np.concatenate([x[1].reshape((1, -1)) for x in test_loss_13], axis=0)

    pos_loss_14 = np.concatenate([x[0].reshape((1, -1)) for x in test_loss_14], axis=0)
    tang_loss_14 = np.concatenate([x[1].reshape((1, -1)) for x in test_loss_14], axis=0)

    pos_loss_15 = np.concatenate([x[0].reshape((1, -1)) for x in test_loss_15], axis=0)
    tang_loss_15 = np.concatenate([x[1].reshape((1, -1)) for x in test_loss_15], axis=0)

    # Output losses to files
    np.savetxt("output/data_size/pos_loss_13.dat", pos_loss_13, delimiter=",")
    np.savetxt("output/data_size/tang_loss_13.dat", tang_loss_13, delimiter=",")

    np.savetxt("output/data_size/pos_loss_14.dat", pos_loss_14, delimiter=",")
    np.savetxt("output/data_size/tang_loss_14.dat", tang_loss_14, delimiter=",")

    np.savetxt("output/data_size/pos_loss_15.dat", pos_loss_15, delimiter=",")
    np.savetxt("output/data_size/tang_loss_15.dat", tang_loss_15, delimiter=",")
