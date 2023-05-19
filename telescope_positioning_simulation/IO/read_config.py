from typing import Union
import yaml
import os


class ReadConfig:
    def __init__(
        self, observator_configuration: Union[None, str] = None, survey: bool = False
    ) -> None:
        self.survey = survey
        if observator_configuration is not None:
            config = ReadConfig.load_yaml(observator_configuration)

        else:
            config = {}

        self.config = self.add_defaults(config)

    @staticmethod
    def load_yaml(config_path: str):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def add_defaults(self, current_config: dict):
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
