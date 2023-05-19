import yaml
import json
import pandas as pd
import os

import numpy as np
import datetime as dt


class SaveSimulation:
    def __init__(self, survey_instance, survey_results) -> None:

        assert survey_instance.save_config is not None
        self.survey_instance = survey_instance
        self.survey_results = survey_results

        save_id = SaveSimulation.generate_run_id()
        self.save_path = f"{os.path.abspath(survey_instance.save_config.rstrip('/'))}/survey_{save_id}"

        os.makedirs(self.save_path)

    def save_results(self):
        result_path = f"{self.save_path}/survey_results.json"

        format_result = {
            index: {
                key: self.survey_results[index][key].tolist()
                for key in self.survey_results[index].keys()
            }
            for index in self.survey_results.keys()
        }

        with open(result_path, "w") as f:
            json.dump(format_result, f)

    def save_config(self):
        formated_config = {
            **self.survey_instance.survey_config,
            **self.survey_instance.telescope_config,
        }
        formated_config["run_id"] = self.save_path.split("/")[0]

        config_path = f"{self.save_path}/run_config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(formated_config, f)

    @staticmethod
    def generate_run_id(random_digits=4):
        _rint = np.random.randint(10**random_digits)
        date_string = (
            str(dt.datetime.now())
            .split(".")[0]
            .replace(" ", "_")
            .replace("-", "_")
            .replace(":", "_")
        )
        return f"{date_string}_{str(_rint).zfill(random_digits)}"

    def __call__(self):
        self.save_results()
        self.save_config()
