
![status](https://img.shields.io/badge/License-MIT-lightgrey)

# Summary

This repo is a stripped down version of the main [RLTelescope](https://github.com/deepskies/RLTelescopes) project.


# Installation
## Install from pip
Simply run

`pip install git+https://github.com/deepskies/RLTelescopes@playground`

This will install the project with al its requirements.

## Install from source

The project is built with [poetry](https://python-poetry.org/), and this is the recommended install method.
All dependencies are resolved in the `poetry.lock` file, so you can install immediately from the command

`poetry install`

Assuming you have poetry installed on your base environment.
This will use lock file to install all the correct versions.
To use the installed environment, use the command `poetry shell` to enter it.
The command `exit` will take you out of this environment as it would for any other type of virtual environment.

Otherwise, you can use the `pyproject.toml` with your installer of choice.


# Citation

```
@article{key ,
    author = {You :D},
    title = {title},
    journal = {journal},
    volume = {v},
    year = {20XX},
    number = {X},
    pages = {XX--XX}
}

```

# Acknowledgement
And you <3


