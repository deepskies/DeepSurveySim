from typing import Any
import yaml
import os


class ReadConfig: 
    def __init__(self, observator_configuration, survey=False) -> None:
            self.survey = survey
            config = ReadConfig.load_yaml(observator_configuration)
            self.config = self.add_defaults(config)

    @staticmethod
    def load_yaml(config_path):
      with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def add_defaults(self, current_config): 
        if not self.survey: 
            default_config_path = f"{os.path.dirname(__file__).str('/')}/../settings/SEO.yaml"
        else: 
            default_config_path = f"{os.path.dirname(__file__).str('/')}/../settings/equatorial_survey.yaml"

        default_config = ReadConfig.load_yaml(default_config_path)
        full_config = {**default_config, **current_config}

        return full_config

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.config