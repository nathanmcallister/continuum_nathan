#!/bin/python3
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from ANN import Model, Dataset, PoseLoss
from utils_data import DataContainer
from typing import List, Tuple


def train(
    dataset: Dataset,
    hidden_layers: List[float],
    iterations: int = 5,
) -> Tuple[List[Model], List[Tuple[float, float]]]:

    models = []
    losses = []
    for i in range(iterations):
        print(f"Size {hidden_layers} | Model {i+1}")
        model_string = "_".join([str(x) for x in hidden_layers])
        model = Model(
            8,
            6,
            hidden_layers,
            loss=PoseLoss(),
            save_path=f"models/model_size/{model_string}_{i}.pt",
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

    train_dataset = Dataset("./training_data/14_2024_07_31_08_55_51.dat")

    # Train models
    models_32, loss_32 = train(train_dataset, [32, 32])
    models_64, loss_64 = train(train_dataset, [64, 64])
    models_128, loss_128 = train(train_dataset, [128, 128])

    # Extract losses
    train_loss_32 = np.array([x[0] for x in loss_32])
    validation_loss_32 = np.array([x[1] for x in loss_32])

    train_loss_64 = np.array([x[0] for x in loss_64])
    validation_loss_64 = np.array([x[1] for x in loss_64])

    train_loss_128 = np.array([x[0] for x in loss_128])
    validation_loss_128 = np.array([x[1] for x in loss_128])

    # Output losses to files
    np.savetxt("output/model_size/train_loss_32.dat", train_loss_32, delimiter=",")
    np.savetxt(
        "output/model_size/validation_loss_32.dat", validation_loss_32, delimiter=","
    )

    np.savetxt("output/model_size/train_loss_64.dat", train_loss_64, delimiter=",")
    np.savetxt(
        "output/model_size/validation_loss_64.dat", validation_loss_64, delimiter=","
    )

    np.savetxt("output/model_size/train_loss_128.dat", train_loss_128, delimiter=",")
    np.savetxt(
        "output/model_size/validation_loss_128.dat", validation_loss_128, delimiter=","
    )

    # Test models
    test_loss_32 = test(test_deltas, test_pos, test_tang, models_32)
    test_loss_64 = test(test_deltas, test_pos, test_tang, models_64)
    test_loss_128 = test(test_deltas, test_pos, test_tang, models_128)

    # Extract losses
    pos_loss_32 = np.concatenate([x[0].reshape((1, -1)) for x in test_loss_32], axis=0)
    tang_loss_32 = np.concatenate([x[1].reshape((1, -1)) for x in test_loss_32], axis=0)

    pos_loss_64 = np.concatenate([x[0].reshape((1, -1)) for x in test_loss_64], axis=0)
    tang_loss_64 = np.concatenate([x[1].reshape((1, -1)) for x in test_loss_64], axis=0)

    pos_loss_128 = np.concatenate(
        [x[0].reshape((1, -1)) for x in test_loss_128], axis=0
    )
    tang_loss_128 = np.concatenate(
        [x[1].reshape((1, -1)) for x in test_loss_128], axis=0
    )

    # Output losses to files
    np.savetxt("output/model_size/pos_loss_32.dat", pos_loss_32, delimiter=",")
    np.savetxt("output/model_size/tang_loss_32.dat", tang_loss_32, delimiter=",")

    np.savetxt("output/model_size/pos_loss_64.dat", pos_loss_64, delimiter=",")
    np.savetxt("output/model_size/tang_loss_64.dat", tang_loss_64, delimiter=",")

    np.savetxt("output/model_size/pos_loss_128.dat", pos_loss_128, delimiter=",")
    np.savetxt("output/model_size/tang_loss_128.dat", tang_loss_128, delimiter=",")
