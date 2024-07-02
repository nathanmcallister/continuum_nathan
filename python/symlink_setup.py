#!/bin/python3
import os
from pathlib import Path

"""
symlink_setup.py
Nathan McAllister - 07/01/2024

Creates all symlinked files, using source files from the 'python' folder
run as administrator once to set up all symlinked files
"""

continuum_nathan = Path(
    __file__
).parent.parent  # change this if file location of this script changes
print(f"continuum_nathan file path: {continuum_nathan}")

# starting with the continuum_arduino.py links

cont_arduino_python = continuum_nathan.joinpath(
    "python", "continuum_arduino.py"
)  # source path
print(f"source file path: {cont_arduino_python}")

try:
    os.remove(
        continuum_nathan.joinpath(
            "experiments", "camarillo_fitting", "continuum_arduino.py"
        )
    )
finally:
    os.symlink(
        cont_arduino_python,
        continuum_nathan.joinpath(
            "experiments", "camarillo_fitting", "continuum_arduino.py"
        ),
    )

try:
    os.remove(
        continuum_nathan.joinpath(
            "experiments", "constant_curvature", "continuum_arduino.py"
        )
    )
finally:
    os.symlink(
        cont_arduino_python,
        continuum_nathan.joinpath(
            "experiments", "constant_curvature", "continuum_arduino.py"
        ),
    )

try:
    os.remove(
        continuum_nathan.joinpath(
            "experiments", "model_control", "continuum_arduino.py"
        )
    )
finally:
    os.symlink(
        cont_arduino_python,
        continuum_nathan.joinpath(
            "experiments", "model_control", "continuum_arduino.py"
        ),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "motor_babble", "continuum_arduino.py")
    )
finally:
    os.symlink(
        cont_arduino_python,
        continuum_nathan.joinpath(
            "experiments", "motor_babble", "continuum_arduino.py"
        ),
    )

try:
    os.remove(
        continuum_nathan.joinpath(
            "experiments", "repeatability", "continuum_arduino.py"
        )
    )
finally:
    os.symlink(
        cont_arduino_python,
        continuum_nathan.joinpath(
            "experiments", "repeatability", "continuum_arduino.py"
        ),
    )

try:
    os.remove(continuum_nathan.joinpath("experiments", "sweep", "continuum_arduino.py"))
finally:
    os.symlink(
        cont_arduino_python,
        continuum_nathan.joinpath("experiments", "sweep", "continuum_arduino.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "tensioning", "continuum_arduino.py")
    )
finally:
    os.symlink(
        cont_arduino_python,
        continuum_nathan.joinpath("experiments", "tensioning", "continuum_arduino.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("python", "servo_setup", "continuum_arduino.py")
    )
finally:
    os.symlink(
        cont_arduino_python,
        continuum_nathan.joinpath("python", "servo_setup", "continuum_arduino.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("python", "spine_setup", "continuum_arduino.py")
    )
finally:
    os.symlink(
        cont_arduino_python,
        continuum_nathan.joinpath("python", "spine_setup", "continuum_arduino.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath(
            "testing", "continuum_arduino", "continuum_arduino.py"
        )
    )
finally:
    os.symlink(
        cont_arduino_python,
        continuum_nathan.joinpath(
            "testing", "continuum_arduino", "continuum_arduino.py"
        ),
    )

# continuum_aurora.py links

cont_aurora_python = continuum_nathan.joinpath(
    "python", "continuum_aurora.py"
)  # source path
print(f"source file path: {cont_aurora_python}")

# deleting existing files
try:
    os.remove(
        continuum_nathan.joinpath("experiments", "aurora_timing", "continuum_aurora.py")
    )
finally:
    os.symlink(
        cont_aurora_python,
        continuum_nathan.joinpath(
            "experiments", "aurora_timing", "continuum_aurora.py"
        ),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "base_position", "continuum_aurora.py")
    )
finally:
    os.symlink(
        cont_aurora_python,
        continuum_nathan.joinpath(
            "experiments", "base_position", "continuum_aurora.py"
        ),
    )
try:
    os.remove(
        continuum_nathan.joinpath(
            "experiments", "camarillo_fitting", "continuum_aurora.py"
        )
    )
finally:
    os.symlink(
        cont_aurora_python,
        continuum_nathan.joinpath(
            "experiments", "camarillo_fitting", "continuum_aurora.py"
        ),
    )
try:
    os.remove(
        continuum_nathan.joinpath(
            "experiments", "constant_curvature", "continuum_aurora.py"
        )
    )
finally:
    os.symlink(
        cont_aurora_python,
        continuum_nathan.joinpath(
            "experiments", "constant_curvature", "continuum_aurora.py"
        ),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "model_control", "continuum_aurora.py")
    )
finally:
    os.symlink(
        cont_aurora_python,
        continuum_nathan.joinpath(
            "experiments", "model_control", "continuum_aurora.py"
        ),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "motor_babble", "continuum_aurora.py")
    )
finally:
    os.symlink(
        cont_aurora_python,
        continuum_nathan.joinpath("experiments", "motor_babble", "continuum_aurora.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "repeatability", "continuum_aurora.py")
    )
finally:
    os.symlink(
        cont_aurora_python,
        continuum_nathan.joinpath(
            "experiments", "repeatability", "continuum_aurora.py"
        ),
    )

try:
    os.remove(continuum_nathan.joinpath("experiments", "sweep", "continuum_aurora.py"))
finally:
    os.symlink(
        cont_aurora_python,
        continuum_nathan.joinpath("experiments", "sweep", "continuum_aurora.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "tensioning", "continuum_aurora.py")
    )
finally:
    os.symlink(
        cont_aurora_python,
        continuum_nathan.joinpath("experiments", "tensioning", "continuum_aurora.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("testing", "continuum_aurora", "continuum_aurora.py")
    )

finally:
    os.symlink(
        cont_aurora_python,
        continuum_nathan.joinpath("testing", "continuum_aurora", "continuum_aurora.py"),
    )

# kinamatics.py links

kinematics_python = continuum_nathan.joinpath("python", "kinematics.py")  # source path
print(f"source file path: {kinematics_python}")

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "aurora_timing", "kinematics.py")
    )
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "aurora_timing", "kinematics.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "base_position", "kinematics.py")
    )
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "base_position", "kinematics.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "camarillo_fitting", "kinematics.py")
    )
except:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "camarillo_fitting", "kinematics.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "constant_curvature", "kinematics.py")
    )
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "constant_curvature", "kinematics.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "model_control", "kinematics.py")
    )
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "model_control", "kinematics.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "model_learning", "kinematics.py")
    )
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "model_learning", "kinematics.py"),
    )

try:
    os.remove(continuum_nathan.joinpath("experiments", "motor_babble", "kinematics.py"))
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "motor_babble", "kinematics.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "new_pivot_cal", "kinematics.py")
    )
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "new_pivot_cal", "kinematics.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "repeatability", "kinematics.py")
    )
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "repeatability", "kinematics.py"),
    )

try:
    os.remove(continuum_nathan.joinpath("experiments", "sweep", "kinematics.py"))
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "sweep", "kinematics.py"),
    )

try:
    os.remove(continuum_nathan.joinpath("experiments", "tensioning", "kinematics.py"))
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("experiments", "tensioning", "kinematics.py"),
    )

try:
    os.remove(continuum_nathan.joinpath("testing", "continuum_aurora", "kinematics.py"))
finally:
    os.symlink(
        kinematics_python,
        continuum_nathan.joinpath("testing", "continuum_aurora", "kinematics.py"),
    )

# utils_data.py links

utils_data_python = continuum_nathan.joinpath("python", "utils_data.py")  # source path
print(f"source file path: {utils_data_python}")

# deleting existing files
try:
    os.remove(
        continuum_nathan.joinpath("experiments", "base_position", "utils_data.py")
    )

finally:
    os.symlink(
        utils_data_python,
        continuum_nathan.joinpath("experiments", "base_position", "utils_data.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "camarillo_fitting", "utils_data.py")
    )
finally:
    os.symlink(
        utils_data_python,
        continuum_nathan.joinpath("experiments", "camarillo_fitting", "utils_data.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "constant_curvature", "utils_data.py")
    )
finally:
    os.symlink(
        utils_data_python,
        continuum_nathan.joinpath("experiments", "constant_curvature", "utils_data.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "model_control", "utils_data.py")
    )
finally:
    os.symlink(
        utils_data_python,
        continuum_nathan.joinpath("experiments", "model_control", "utils_data.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "model_learning", "utils_data.py")
    )
finally:
    os.symlink(
        utils_data_python,
        continuum_nathan.joinpath("experiments", "model_learning", "utils_data.py"),
    )

try:
    os.remove(continuum_nathan.joinpath("experiments", "motor_babble", "utils_data.py"))
finally:
    os.symlink(
        utils_data_python,
        continuum_nathan.joinpath("experiments", "motor_babble", "utils_data.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "new_pivot_cal", "utils_data.py")
    )
finally:
    os.symlink(
        utils_data_python,
        continuum_nathan.joinpath("experiments", "new_pivot_cal", "utils_data.py"),
    )

try:
    os.remove(
        continuum_nathan.joinpath("experiments", "repeatability", "utils_data.py")
    )
finally:
    os.symlink(
        utils_data_python,
        continuum_nathan.joinpath("experiments", "repeatability", "utils_data.py"),
    )

try:
    os.remove(continuum_nathan.joinpath("experiments", "sweep", "utils_data.py"))
finally:
    os.symlink(
        utils_data_python,
        continuum_nathan.joinpath("experiments", "sweep", "utils_data.py"),
    )

try:
    os.remove(continuum_nathan.joinpath("testing", "continuum_aurora", "utils_data.py"))
finally:
    os.symlink(
        utils_data_python,
        continuum_nathan.joinpath("testing", "continuum_aurora", "utils_data.py"),
    )
