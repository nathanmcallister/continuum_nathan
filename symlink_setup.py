import os
from pathlib import Path

''' creates all symlinked file, using source files from the 'python' folder '''
''' run once to set up all symlinked files '''

# starting with the continuum_arduino links

continuum_nathan = Path(__file__).parent
print(f"continuum_nathan file path: {continuum_nathan}")

cont_arduino_python = continuum_nathan.joinpath("python", "continuum_arduino.py") # source path
print(f"source file path: {cont_arduino_python}")

''' UNCOMMENT BELOW WHEN READY TO RUN '''
# os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "camarillo_fitting", "continuum_arduino.py"))
# os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "constant_curvature", "continuum_arduino.py"))
# os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "model_control", "continuum_arduino.py"))
# os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "motor_babble", "continuum_arduino.py"))
# os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "repeatability", "continuum_arduino.py"))
# os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "sweep", "continuum_arduino.py"))
# os.symlink(cont_arduino_python, continuum_nathan.joinpath("experiments", "tensioning", "continuum_arduino.py"))
# os.symlink(cont_arduino_python, continuum_nathan.joinpath("testing", "continuum_arduino", "continuum_arduino.py"))
# os.symlink(cont_arduino_python, continuum_nathan.joinpath("testing", "continuum_arduino", "continuum_arduino.py"))

''' still need to add symlink commands for other files '''

