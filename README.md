
![status](https://img.shields.io/badge/License-MIT-lightgrey)
# Summary
This branch contains an initial in progress adaptation of the repository to support the training, tuning, and testing of RL algorithms using the RLLib library.

# Installation
## Install from pip
Creare a new environment with python 3.9.0.
For example, using conda:<br>
`conda create --name myenv python=3.9.0`<br>
Install the main project requirements:<br>
`pip install git+https://github.com/deepskies/TelescopePositioningSimulation`
<br>
Install the requirements needed to use RLLib for training, tuning and testing, using the requirements file in the main project's folder:<br>
`pip install -r requirements.txt`

## Install from source
The project is built with [poetry](https://python-poetry.org/), and this is the recommended install method.
All dependencies are resolved in the `poetry.lock` file, so you can install immediately from the command.
```
poetry shell
poetry install

```
Assuming you have poetry installed on your base environment.
This will use lock file to install all the correct versions.
To use the installed environment, use the command `poetry shell` to enter it.
The command `exit` will take you out of this environment as it would for any other type of virtual environment.
Otherwise, you can use the `pyproject.toml` with your installer of choice.
To verify all the depedencies are properly installed - run `python run pytest`.
<br>
Install the requirements needed to use RLLib for training, tuning and testing, using the requirements file in the main project's folder:<br>
`pip install -r requirements.txt`

# Training
The only supported algorithm for this branch is PPO.<br>
Other algorithms can be integrated based on RLLib.<br>
Before training, modify the content of the configuration files <i>experiment_config.yaml</i> and <i>default_config.yaml</i>.<br>
To train the PPO algorithm using RLLib on the online environment, navigate to the telescope_positioning_folder and run:<br>
`python model_train.py -m train -a PPO`<br>
To tune the algorithm according to a range of hyper-parameters to be defined in the <i>config.py</i> file, run:<br>
`python model_train.py -m tune -a PPO`<br>
Once run the tuning module, checkpoints will be generated in the results folder, paired with tensorboard runs that can be visualized running in that folder the tensorboard command:<br>
`tensorboard --logdir=.`<br>
A web server will be exposed to <i>localhost:6006</i> to visualize the results.




