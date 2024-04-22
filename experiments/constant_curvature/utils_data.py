import datetime
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class DataContainer:
    date: Tuple[int, int, int] = None
    time: Tuple[int, int, int] = None
    num_cables: int = None
    num_measurements: int = None
    inputs: List[np.ndarray] = field(default_factory=list)
    outputs: List[np.ndarray] = field(default_factory=list)
    num_auroras: int = 1
    prefix: str = "data"

    def file_export(self, filename: str = None):
        if not filename:
            filename = self.prefix + f"_{self.date[0]:02n}_{self.date[1]:02n}_{self.date[2]:02n}_{self.time[0]:02n}_{self.time[1]:02n}_{self.time[2]:02n}.dat"

        with open(filename, "w") as file:
            file.write(f"DATE: {self.date[0]}-{self.date[1]}-{self.date[2]}\n")
            file.write(f"TIME: {self.time[0]}-{self.time[1]}-{self.time[2]}\n")
            file.write(f"NUM_CABLES: {self.num_cables}\n")
            file.write(f"NUM_AURORAS: {self.num_auroras}\n")
            file.write(f"NUM_MEASUREMENTS: {self.num_measurements}\n")
            file.write("---\n")

            counter = 0
            for input, output in zip(self.inputs, self.outputs):
                file.write(f"{counter},")

                for input_val in input:
                    file.write(
                        f"{input_val},"
                    )

                for output_val in output:
                    if output_val != output[-1]:
                        file.write(f"{output_val},")
                    else:
                        file.write(f"{output_val}\n")

                counter += 1

    def file_import(self, filename: str):
        with open(filename, "r") as file:
            date_line = file.readline()
            date_list = date_line.split(":")
            assert date_list[0] == "DATE"

            self.date = tuple([int(x) for x in date_list[1].split("-")])

            time_line = file.readline()
            time_list = time_line.split(":")
            assert time_list[0] == "TIME"

            self.time = tuple([int(x) for x in time_list[1].split("-")])

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

            num_outputs = 6 * self.num_auroras

            while line := file.readline():
                row = line.split(",")
                self.inputs.append(
                    np.array(
                        [float(x) for x in row[1 : self.num_cables + 1]],
                        dtype=float,
                    )
                )
                self.outputs.append(
                    np.array(
                        [float(x) for x in row[self.num_cables + 1 :]],
                        dtype=float,
                    )
                )

            assert len(self.inputs) == len(self.outputs) == self.num_measurements

    def from_raw_data(
        self,
        date: Tuple[int, int, int],
        time: Tuple[int, int, int],
        num_cables: int,
        num_measurements: int,
        cable_deltas: np.ndarray,
        positions: np.ndarray,
        orientations: np.ndarray,
        num_auroras: int = 1,
    ):
        assert (cable_deltas.shape == (num_cables, num_measurements))
        assert (positions.shape == (3 * num_auroras, num_measurements))
        assert (orientations.shape == (3 * num_auroras, num_measurements))

        self.date = date
        self.time = time
        self.num_cables = num_cables
        self.num_measurements = num_measurements
        self.num_auroras = num_auroras

        self.inputs = []
        self.outputs = []

        input_size = num_cables
        output_size = num_auroras * 6

        for i in range(num_measurements):
            self.inputs.append(cable_deltas[:, i].flatten())
            self.outputs.append(np.concatenate([positions[:, i], orientations[:, i]], axis=0).flatten())

    def set_date_and_time(self):
        now = datetime.datetime.now()

        self.date = (now.year, now.month, now.day)
        self.time = (now.hour, now.minute, now.second)


def get_header_str(num_cables: int, num_measurements: int, num_auroras: int = 1) -> str:
    now = datetime.datetime.now()
    date_str = "DATE: " + now.strftime("%Y-%m-%d") + "\n"
    time_str = "TIME: " + now.strftime("%H:%M:%S") + "\n"
    num_cables_str = "NUM_CABLES: " + str(num_cables) + "\n"
    num_auroras_str = "NUM_AURORAS: 1" + "\n"
    num_measurements_str = "NUM_MEASUREMENTS: " + str(num_measurements) + "\n"

    return (
        date_str
        + time_str
        + num_cables_str
        + num_auroras_str
        + aurora_dofs_str
        + num_measurements_str
        + "---\n"
    )


def export_file(
    header_str: str,
    cable_deltas: np.ndarray,
    positions: np.ndarray,
    orientations: np.ndarray,
    filename: str,
) -> None:

    with open(filename, "w") as file:
        file.write(header_str)

        for i in range(positions.shape[1]):
            file.write(str(i) + ",")
            for delta in cable_deltas[:, i]:
                file.write(str(delta) + ",")
            for pos in positions[:, i]:
                file.write(str(pos) + ",")
            for tang in orientations[:, i]:
                if tang != orientations[2, i]:
                    file.write(str(tang) + ",")
                else:
                    file.write(str(tang))

            file.write("\n")


def load_file(filename: str) -> np.ndarray:

    with open(file_name, "r") as file:
        date_line = file.readline()
        date_list = date_line.split(":")
        assert date_list[0] == "DATE"

        date = tuple([int(x) for x in date_list[1].split("-")])

        time_line = file.readline()
        time_list = time_line.split(":")
        assert time_list[0] == "TIME"

        time = tuple([int(x) for x in time_list[1:]])

        num_cables_line = file.readline()
        num_cables = int(num_cables_line.split(":")[1])

        num_auroras_line = file.readline()
        num_auroras_list = num_auroras_line.split(":")
        num_auroras = int(num_auroras_list[1])

        aurora_dofs_line = file.readline()
        aurora_dofs_list = aurora_dofs_line.split(":")
        aurora_dofs_list[0] == "AURORA_DOFS"

        aurora_dofs = [int(x) for x in aurora_dofs_list[1].split(",")]
        assert num_auroras == len(aurora_dofs)

        num_measurements_line = file.readline()
        num_measurements_list = num_measurements_line.split(":")
        assert num_measurements_list[0] == "NUM_MEASUREMENTS"

        num_measurements = int(num_measurements_list[1])

        spacer = file.readline()
        assert spacer.strip() == "---"

        num_outputs = 0
        for dof in aurora_dofs:
            if dof == 5:
                num_outputs += 5
            else:
                num_outputs += 6

        inputs = []
        outputs = []
        while line := file.readline():
            row = line.split(",")
            inputs.append(
                np.array(
                    [float(x) for x in row[1 : num_cables + 1]],
                    dtype=float,
                )
            )
            outputs.append(
                np.array(
                    [float(x) for x in row[num_cables + 1 :]],
                    dtype=float,
                )
            )

        assert len(inputs) == len(outputs) == num_measurements

    return DataContainer(date, time, num_cables, num_measurements, inputs, outputs)
