# Continuum Robot Development

This repository contains all of the code for the Halter Lab's continuum robot development.  Everything from registration, communication with the Aurora and Arduino, constant curvature models, training and testing of models, and controller development can be found in the repository.

All necessary Python libraries can be found in requirements.txt.  They can be installed using the following command:
```bash
pip install -r requirements.txt
```

The thesis associated with this repository can be read [here](https**://drive.google.com/file/d/1uIR_xviVqUKbwXYV8wjdk6yPhhzn69-D/view?usp=sharing).

The structure of the repository is as follows:
 - **Arduino**: Contains all code for the Arduino

 - **Data**: All raw data from the Aurora is stored here.  Also contains a script to sync with the Halter-Puppet

 - **Experiments**: Contains all of the scripts associated with each experiment and its figures.  The experiments are:
   - **Aurora timing**: Finding how long communication with the Aurora takes
   - **Base position**: Locating the position of the base (and therefore spine) within the registered frame
   - **Camarillo fitting**: Used to determine the Caramillo stiffness parameters $k_a, k_b$ from data.
   - **Composite photo**: Showing how the constant curvature model departs from reality by creating a composite photo containing the spine bending different amounts
   - **Constant curvature**: Testing the performance of the Mike and Camarillo constant curvature models.
   - **Hysteresis replication**: Replicating the distribution of kinematic babble points using slack cables and hysteresis with the Camarillo constant curvature model.
   - **Model comparison**: Used to compare the constant curvature and learned models.
   - **Model control**: Using learned models in the open and closed loop control of the robot.
   - **Model learning**: Training of models on data collected from the spine to develop a kinematics model.
   - **Motor babble**: Apply a random uniform distribution of inputs to the robot to collect training data.
   - **Multi-input control**: Control using models that use the current and previous cable displacements.
   - **Multi-input learning**: Training models that use current and previous cable displacements.
   - **Penprove analysis**: Determine the effects of pivot calibration and its accuracy.
   - **Point mesh**: Develop a mesh of the workspace of the robot.
   - **Repeatability**: Determine how the starting position affects the spread of points for a given command.
   - **SPIE demo**: Combining EIT and the continuum robot.
   - **Sweep**: Sweep the robot around the workspace to see how position changes with cable displacements.
   - **Tensioning**: How to tension the cables in the robot.
   - **Two-segment learning**: Improvements to the two-segment models in training.
 
 - **Matlab**: Contains all Matlab code
 
 - **Python**: Contains all Python code
   - **Learning**: All classes and functions needed to train a model.
   - **Modeling**: Contains Mike and Camarillo constant curvature models, and all utils needed to run them.
   - **Continuum Arduino**: Interaction with the Arduino, used to send motor commands.
   - **Continuum Aurora**: Interaction with the Aurora, used to receive tip transforms.
   - **Continuum Control**: Open and closed loop controllers of the robot.
   - **Kinematics**: Useful calculations/ functions for kinematic modeling.
   - **Pivot Cal RANSAC**: Performs pivot calibration of a pen probe using RANSAC to remove outliers.
   - **Rigid registration**: Used to register model and tip frames, generating necessary transformations.
   - **Symlink setup**: This function creates all symlinks in the experiment, setup, and testing folders by linking contents found in a symlinks.txt text file.
   - **Utils data**: Used for saving and importing data from the robot.
 
 - **Setup**: Scripts used to setup the robot.
 
 - **Testing**: Used to test the different scripts in the Python folder.

 - **Tools**: Contains all transformations and pen probes.
 
 - **Utils**: Other miscellaneous utilities.
