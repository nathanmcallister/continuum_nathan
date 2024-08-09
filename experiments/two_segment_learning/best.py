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
            [128, 128],
            loss=PoseLoss(),
            save_path=f"models/best/{i}.pt",
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

    dataset = Dataset("./training_data/15_2024_07_31_08_56_59.dat")

    # Train models
    models, loss = train(dataset)

    # Extract losses
    train_loss = np.array([x[0] for x in loss])
    validation_loss = np.array([x[1] for x in loss])

    # Output losses to files
    np.savetxt("output/best/train_loss.dat", train_loss, delimiter=",")
    np.savetxt("output/best/validation_loss.dat", validation_loss, delimiter=",")

    # Test models
    test_loss = test(test_deltas, test_pos, test_tang, models)

    # Extract losses
    pos_loss = np.concatenate([x[0].reshape((1, -1)) for x in test_loss], axis=0)
    tang_loss = np.concatenate([x[1].reshape((1, -1)) for x in test_loss], axis=0)

    # Output losses to files
    np.savetxt("output/best/pos_loss.dat", pos_loss, delimiter=",")
    np.savetxt("output/best/tang_loss.dat", tang_loss, delimiter=",")
