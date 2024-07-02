import os
from pathlib import Path

''' creates all symlinked file, using source files from the 'python' folder '''
''' run as administrator once to set up all symlinked files '''

continuum_nathan = Path(__file__).parent.parent # change this if file location of this script changes
print(f"continuum_nathan file path: {continuum_nathan}")

# starting with the continuum_arduino.py links

cont_arduino_python = continuum_nathan.joinpath("python", "continuum_arduino.py") # source path
print(f"source file path: {cont_arduino_python}")

# deleting existing files
os.remove(continuum_nathan.joinpath("experiments", "camarillo_fitting", "continuum_arduino.py"))
os.remove(continuum_nathan.joinpath("experiments", "constant_curvature", "continuum_arduino.py"))
os.remove(continuum_nathan.joinpath("experiments", "model_control", "continuum_arduino.py"))
os.remove(continuum_nathan.joinpath("experiments", "motor_babble", "continuum_arduino.py"))
os.remove(continuum_nathan.joinpath("experiments", "repeatability", "continuum_arduino.py"))
os.remove(continuum_nathan.joinpath("experiments", "sweep", "continuum_arduino.py"))
os.remove(continuum_nathan.joinpath("experiments", "tensioning", "continuum_arduino.py"))
os.remove(continuum_nathan.joinpath("python", "servo_setup", "continuum_arduino.py"))
os.remove(continuum_nathan.joinpath("python", "spine_setup", "continuum_arduino.py"))
os.remove(continuum_nathan.joinpath("testing", "continuum_arduino", "continuum_arduino.py"))

# creating symlinked files
os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "camarillo_fitting", "continuum_arduino.py"))
os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "constant_curvature", "continuum_arduino.py"))
os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "model_control", "continuum_arduino.py"))
os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "motor_babble", "continuum_arduino.py"))
os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "repeatability", "continuum_arduino.py"))
os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "sweep", "continuum_arduino.py"))
os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "tensioning", "continuum_arduino.py"))
os.symlink(cont_arduino_python, continuum_nathan.joinpath("python", "servo_setup", "continuum_arduino.py"))
os.symlink(cont_arduino_python, continuum_nathan.joinpath("python", "spine_setup", "continuum_arduino.py"))
os.symlink(cont_arduino_python, continuum_nathan.joinpath("testing", "continuum_arduino", "continuum_arduino.py"))

# continuum_aurora.py links

cont_aurora_python = continuum_nathan.joinpath("python", "continuum_aurora.py") # source path
print(f"source file path: {cont_aurora_python}")

# deleting existing files
os.remove(continuum_nathan.joinpath("experiments", "aurora_timing", "continuum_aurora.py"))
os.remove(continuum_nathan.joinpath("experiments", "base_position", "continuum_aurora.py"))
os.remove(continuum_nathan.joinpath("experiments", "camarillo_fitting", "continuum_aurora.py"))
os.remove(continuum_nathan.joinpath("experiments", "constant_curvature", "continuum_aurora.py"))
os.remove(continuum_nathan.joinpath("experiments", "model_control", "continuum_aurora.py"))
os.remove(continuum_nathan.joinpath("experiments", "motor_babble", "continuum_aurora.py"))
os.remove(continuum_nathan.joinpath("experiments", "repeatability", "continuum_aurora.py"))
os.remove(continuum_nathan.joinpath("experiments", "sweep", "continuum_aurora.py"))
os.remove(continuum_nathan.joinpath("experiments", "tensioning", "continuum_aurora.py"))
os.remove(continuum_nathan.joinpath("testing", "continuum_aurora", "continuum_aurora.py"))

# creating symlinked files
os.symlink(cont_aurora_python, continuum_nathan.joinpath("experiments", "aurora_timing", "continuum_aurora.py"))
os.symlink(cont_aurora_python, continuum_nathan.joinpath("experiments", "base_position", "continuum_aurora.py"))
os.symlink(cont_aurora_python, continuum_nathan.joinpath("experiments", "camarillo_fitting", "continuum_aurora.py"))
os.symlink(cont_aurora_python, continuum_nathan.joinpath("experiments", "constant_curvature", "continuum_aurora.py"))
os.symlink(cont_aurora_python, continuum_nathan.joinpath("experiments", "model_control", "continuum_aurora.py"))
os.symlink(cont_aurora_python, continuum_nathan.joinpath("experiments", "motor_babble", "continuum_aurora.py"))
os.symlink(cont_aurora_python, continuum_nathan.joinpath("experiments", "repeatability", "continuum_aurora.py"))
os.symlink(cont_aurora_python, continuum_nathan.joinpath("experiments", "sweep", "continuum_aurora.py"))
os.symlink(cont_aurora_python, continuum_nathan.joinpath("experiments", "tensioning", "continuum_aurora.py"))
os.symlink(cont_arduino_python, continuum_nathan.joinpath("testing", "continuum_aurora", "continuum_arduino.py"))

# kinamatics.py links

kinematics_python = continuum_nathan.joinpath("python", "kinematics.py") # source path
print(f"source file path: {kinematics_python}")

# deleting existing files
os.remove(continuum_nathan.joinpath("experiments", "aurora_timing", "kinematics.py"))
os.remove(continuum_nathan.joinpath("experiments", "base_position", "kinematics.py"))
os.remove(continuum_nathan.joinpath("experiments", "camarillo_fitting", "kinematics.py"))
os.remove(continuum_nathan.joinpath("experiments", "constant_curvature", "kinematics.py"))
os.remove(continuum_nathan.joinpath("experiments", "model_control", "kinematics.py"))
os.remove(continuum_nathan.joinpath("experiments", "model_learning", "kinematics.py"))
os.remove(continuum_nathan.joinpath("experiments", "motor_babble", "kinematics.py"))
os.remove(continuum_nathan.joinpath("experiments", "new_pivot_cal", "kinematics.py"))
os.remove(continuum_nathan.joinpath("experiments", "repeatability", "kinematics.py"))
os.remove(continuum_nathan.joinpath("experiments", "sweep", "kinematics.py"))
os.remove(continuum_nathan.joinpath("experiments", "tensioning", "kinematics.py"))
os.remove(continuum_nathan.joinpath("testing", "continuum_aurora", "kinematics.py"))

# creating symlinked files
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "aurora_timing", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "base_position", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "camarillo_fitting", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "constant_curvature", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "model_control", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "model_learning", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "motor_babble", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "new_pivot_cal", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "repeatability", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "sweep", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("experiments", "tensioning", "kinematics.py"))
os.symlink(kinematics_python, continuum_nathan.joinpath("testing", "continuum_aurora", "kinematics.py"))

# utils_data.py links

utils_data_python = continuum_nathan.joinpath("python", "utils_data.py") # source path
print(f"source file path: {utils_data_python}")

# deleting existing files
os.remove(continuum_nathan.joinpath("experiments", "base_position", "utils_data.py"))
os.remove(continuum_nathan.joinpath("experiments", "camarillo_fitting", "utils_data.py"))
os.remove(continuum_nathan.joinpath("experiments", "constant_curvature", "utils_data.py"))
os.remove(continuum_nathan.joinpath("experiments", "model_control", "utils_data.py"))
os.remove(continuum_nathan.joinpath("experiments", "model_learning", "utils_data.py"))
os.remove(continuum_nathan.joinpath("experiments", "motor_babble", "utils_data.py"))
os.remove(continuum_nathan.joinpath("experiments", "new_pivot_cal", "utils_data.py"))
os.remove(continuum_nathan.joinpath("experiments", "repeatability", "utils_data.py"))
os.remove(continuum_nathan.joinpath("experiments", "sweep", "utils_data.py"))
os.remove(continuum_nathan.joinpath("testing", "continuum_aurora", "utils_data.py"))

# creating symlinked files
os.symlink(utils_data_python, continuum_nathan.joinpath("experiments", "base_position", "utils_data.py"))
os.symlink(utils_data_python, continuum_nathan.joinpath("experiments", "camarillo_fitting", "utils_data.py"))
os.symlink(utils_data_python, continuum_nathan.joinpath("experiments", "constant_curvature", "utils_data.py"))
os.symlink(utils_data_python, continuum_nathan.joinpath("experiments", "model_control", "utils_data.py"))
os.symlink(utils_data_python, continuum_nathan.joinpath("experiments", "model_learning", "utils_data.py"))
os.symlink(utils_data_python, continuum_nathan.joinpath("experiments", "motor_babble", "utils_data.py"))
os.symlink(utils_data_python, continuum_nathan.joinpath("experiments", "new_pivot_cal", "utils_data.py"))
os.symlink(utils_data_python, continuum_nathan.joinpath("experiments", "repeatability", "utils_data.py"))
os.symlink(utils_data_python, continuum_nathan.joinpath("experiments", "sweep", "utils_data.py"))
os.symlink(utils_data_python, continuum_nathan.joinpath("testing", "continuum_aurora", "utils_data.py"))


''' other symlinked files?? utils_cc.py?? mike_cc.py?? camarillo_cc.py?? '''