import yaml
import os

class ReadConfig:
    def __init__(
        self, experiment_configuration: str = None
    ) -> None:
        if experiment_configuration is not None:
            config = ReadConfig.load_yaml(experiment_configuration)
        else:
            config = {}
        self.config = self.add_defaults(config)

    @staticmethod
    def load_yaml(config_path: str):
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir,config_path)
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def add_defaults(self, current_config: dict):
        default_config_path = 'default_config.yaml'
        default_config = ReadConfig.load_yaml(default_config_path)
        full_config = {**default_config, **current_config}
        return full_config

    def __call__(self):
        return self.config
