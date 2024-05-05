import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import os
import datetime
import utils_data


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        loss: nn.Module = nn.MSELoss(),
        activation: nn.Module = nn.ReLU(),
        output_activation: nn.Module = None,
        lr: float = 1e-3,
        checkpoints_path: str = None,
        save_path: str = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.device = "cpu"
        # self.device = (
        #    "cuda"
        #    if torch.cuda.is_available()
        #    else "mps" if torch.backends.mps.is_available() else "cpu"
        # )
        print(
            f"Using {self.device} device for model (this was set intentionally due to the sizes of the matrices used.  If this needs to be changed, go to line 30 of ANN.py"
        )

        self.model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "input",
                        nn.Linear(
                            self.input_dim,
                            hidden_layers[0],
                        ).double(),
                    ),
                    ("input_activation", activation),
                ]
            )
        )

        for i in range(len(hidden_layers) - 1):
            self.model.add_module(
                f"hidden{i+1}",
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]).double(),
            )
            self.model.add_module(f"activation{i+1}", activation)

        self.model.add_module(
            "output", nn.Linear(hidden_layers[-1], output_dim).double()
        )

        if output_activation:
            self.model.add_module("output_activation", output_activation)

        self.model = self.model.to(self.device)

        self.loss = loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.checkpoints_path = checkpoints_path
        self.save_path = save_path

    # Must send x to self.device
    def forward(self, x):
        return self.model(x)

    def train_epoch(self, dataloader: DataLoader) -> float:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss = 0

        self.model.train()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()

            if batch % 64 == 0:
                current_loss, current = loss.item(), (batch + 1) * len(X)
                print(
                    f"Loss: {current_loss:>7f} [{current:>5d}/{size:>5d}]",
                    flush=True,
                )

        train_loss /= num_batches

        return train_loss

    def test_epoch(self, dataloader: DataLoader) -> float:
        self.model.eval()
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                test_loss += self.loss(pred, y).item()

        test_loss /= num_batches
        print(f"Avg test loss: {test_loss:>7f}", flush=True)

        return test_loss

    def test_dataset(self, dataset: Dataset):
        dataloader = DataLoader(dataset)

        self.model.eval()
        test_loss = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                test_loss.append(self.loss(pred, y).item())

        return test_loss

    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        num_epochs: int = 2048,
        checkpoints: bool = False,
        save_model: bool = False,
    ) -> Tuple[List[float], ...]:
        train_loss = []
        test_loss = []

        if checkpoints:
            now = datetime.datetime.now()

            dir_path = os.path.dirname(os.path.realpath(__file__))

            checkpoints_dir = os.path.join(dir_path, "checkpoints/")

            if not os.path.exists(checkpoints_dir):
                os.mkdir(checkpoints_dir)

            checkpoints_path = f"{now.year}_{now.month:02n}_{now.day:02n}_{now.hour:02n}_{now.minute:02n}_{now.second:02n}/"

            checkpoints_path = os.path.join(checkpoints_dir, checkpoints_path)

            if not os.path.exists(checkpoints_path):
                os.mkdir(checkpoints_path)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------", flush=True)
            train_loss.append(self.train_epoch(train_dataloader))

            if test_dataloader:
                test_loss.append(self.test_epoch(test_dataloader))

            if checkpoints:
                file_path = os.path.join(checkpoints_path + f"epoch_{epoch+1}.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "validation_loss": test_loss[-1],
                        "model_state_dict": self.model.state_dict(),
                        "optim_state_dict": self.optimizer.state_dict(),
                    },
                    file_path,
                )

        if checkpoints:
            min_val_loss = min(enumerate(test_loss), key=lambda x: x[1])
            epoch = min_val_loss[0] + 1

            checkpoint = torch.load(checkpoints_path + f"epoch_{epoch}.pt")
            assert (
                checkpoint["epoch"] == epoch
                and checkpoint["validation_loss"] == min_val_loss[1]
            )

            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Epoch with lowest validation loss (epoch {epoch}) loaded into model"
            )
            self.model.eval()

            if save_model:
                self.save()

        if test_dataloader:
            return train_loss, test_loss

        return train_loss, None

    def save(self, model_save_path: str = None) -> None:
        if model_save_path:
            self.save_path = model_save_path

        assert (
            self.save_path
        ), "No model save path specified in initialization or in save function"
        if not self.save_path.endswith(".pt"):
            self.save_path += ".pt"
        torch.save(self.model.state_dict(), self.save_path)

    def load(self, model_load_path: str) -> None:
        if not model_load_path.endswith(".pt"):
            model_load_path += ".pt"
        self.model.load_state_dict(torch.load(model_load_path))
        self.model.to(self.device)


class Dataset(Dataset):
    def __init__(self, filename: str = None):
        self.date = None
        self.time = None
        self.num_cables = None
        self.num_auroras = None

        self.device = "cpu"
        # self.device = (
        #    "cuda"
        #    if torch.cuda.is_available()
        #    else "mps" if torch.backends.mps.is_available() else "cpu"
        # )
        print(
            f"Using {self.device} device for dataset (this was set intentionally due to the sizes of the matrices used.  If this needs to be changed, go to line 200 of ANN.py"
        )

        self.inputs = []
        self.outputs = []

        if filename:
            self._parse_file(filename)

    def load_from_file(self, filename: str):
        data = utils_data.DataContainer()
        data.file_import(filename)
        self.load_from_DataContainer(data)

    def load_from_DataContainer(self, data: utils_data.DataContainer):
        self.date = data.date
        self.time = data.time
        self.num_cables = data.num_cables
        self.num_auroras = data.num_auroras

        self.inputs = [torch.from_numpy(input).to(self.device) for input in data.inputs]
        self.outputs = [
            torch.from_numpy(output).to(self.device) for output in data.outputs
        ]

    def from_raw(
        self,
        date: Tuple[int, int, int],
        time: Tuple[int, int, int],
        num_cables: int,
        num_auroras: int,
        inputs: List[np.ndarray],
        outputs: List[np.ndarray],
    ):
        self.date = date
        self.time = time
        self.num_cables = num_cables
        self.num_auroras = num_auroras
        self.inputs = inputs
        self.outputs = outputs

    def _parse_file(self, filename: str):
        with open(filename, "r") as file:
            date_line = file.readline()
            date_list = date_line.split(":")
            assert date_list[0] == "DATE"

            self.date = tuple([int(x) for x in date_list[1].split("-")])

            time_line = file.readline()
            time_list = time_line.split(":")
            assert time_list[0] == "TIME"

            self.time = tuple([int(x) for x in time_list[1:]])

            num_cables_line = file.readline()
            self.num_cables = int(num_cables_line.split(":")[1])

            num_auroras_line = file.readline()
            num_auroras_list = num_auroras_line.split(":")
            self.num_auroras = int(num_auroras_list[1])

            num_measurements_line = file.readline()
            num_measurements_list = num_measurements_line.split(":")
            assert num_measurements_list[0] == "NUM_MEASUREMENTS"

            self.num_measurements = int(num_measurements_list[1])

            spacer = file.readline()
            assert spacer.strip() == "---"

            num_outputs = 0
            for dof in self.aurora_dofs:
                if dof == 5:
                    num_outputs += 5
                else:
                    num_outputs += 6

            while line := file.readline():
                row = line.split(",")
                self.inputs.append(
                    torch.tensor(
                        [float(x) for x in row[1 : self.num_cables + 1]],
                        dtype=float,
                    )
                ).to(self.device)
                self.outputs.append(
                    torch.tensor(
                        [float(x) for x in row[self.num_cables + 1 :]],
                        dtype=float,
                    )
                ).to(self.device)

            assert len(self.inputs) == len(self.outputs) == self.num_measurements

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def save(self, filename: str = "dataset_out.txt"):
        raise NotImplementedError


class PoseLoss(nn.Module):
    def __init__(self, scale=10, num_coils=1):
        super(PoseLoss, self).__init__()

        self.device = "cpu"
        # self.device = (
        #    "cuda"
        #    if torch.cuda.is_available()
        #    else "mps" if torch.backends.mps.is_available() else "cpu"
        # )
        print(
            f"Using {self.device} device for PoseLoss (this was set intentionally due to the sizes of the matrices used.  If this needs to be changed, go to line 312 of ANN.py"
        )
        self.num_coils = num_coils
        self.weights = torch.from_numpy(
            np.concatenate((np.ones(3 * num_coils), scale * np.ones(3 * num_coils)))
        ).to(self.device)

    def forward(self, input, target):
        expanded_weights = self.weights.expand(input.size(0), -1)

        return nn.functional.mse_loss(
            input * expanded_weights, target * expanded_weights
        )


class PositionLoss(nn.Module):
    def __init__(self):
        super(PositionLoss, self).__init__()

    def forward(self, input, target):
        return nn.functional.mse_loss(input[:, :3], target[:, :3])


class OrientationLoss(nn.Module):
    def __init__(self):
        super(OrientationLoss, self).__init__()

    def forward(self, input, target):
        return nn.functional.mse_loss(input[:, 3:], target[:, 3:])
