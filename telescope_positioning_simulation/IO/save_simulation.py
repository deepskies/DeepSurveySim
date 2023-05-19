import yaml
import json
import os

import numpy as np
import datetime as dt


class SaveSimulation:
    def __init__(self, survey_instance, survey_results) -> None:

        assert survey_instance.survey.save_config is not None

        self.survey_instance = survey_instance
        self.survey_results = survey_results

        save_id = SaveSimulation.generate_run_id()
        self.save_path = (
            f"{survey_instance.survey.save_config.rstrip('/')}/survey_{save_id}"
        )

    def save_results(self):
        result_path = f"{self.save_path}/survey_results.json"
        with open(result_path, "w") as f:
            json.dump(self.survey_results, f)

    def save_config(self):
        formated_config = {}

        config_path = f"{self.save_path}/run_config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(formated_config, f)

    @staticmethod
    def generate_run_id(random_digits=4):
        _rint = np.random.randint(10**random_digits)
        return f"{dt.now()}_{str(_rint).zfill(random_digits)}"

    def __call__(self):

        os.makedirs(self.save_path)

        self.save_results()
        self.save_config()
