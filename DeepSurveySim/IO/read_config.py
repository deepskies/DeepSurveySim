from typing import Union
import yaml
import os


class ReadConfig:
    """
    Read a .yaml file to a dictionary, and adds defaults

    Args:
        observator_configuration (Union[None, str], optional): Path to configuration file to read. Defaults to None.
        survey (bool, optional): Read a survey configuration, filling in those defaults. Defaults to False.

    Examples:
        >>> observatory_config = IO.ReadConfig(observatory_config_path)()
            survey_config = IO.ReadConfig(survey_config_path, survey=True)()
    """

    def __init__(
        self, observator_configuration: Union[None, str] = None, survey: bool = False
    ) -> None:

        self.survey = survey
        if observator_configuration is not None:
            config = ReadConfig.load_yaml(observator_configuration)

        else:
            config = {}

        self.config = self._add_defaults(config)

    @staticmethod
    def load_yaml(config_path: str):
        """Read a yaml file from a path

        Args:
            config_path (str): path to file, .yaml or .yml.

        Returns:
            dict: Read contents of the config
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _add_defaults(self, current_config: dict):
        if not self.survey:
            default_config_path = (
                f"{os.path.dirname(__file__).rstrip('/')}/../settings/SEO.yaml"
            )
        else:
            default_config_path = f"{os.path.dirname(__file__).rstrip('/')}/../settings/equatorial_survey.yaml"

        default_config = ReadConfig.load_yaml(default_config_path)
        full_config = {**default_config, **current_config}

        return full_config

    def __call__(self):
        return self.config
