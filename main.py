import os
import tempfile
import numpy as np
np.object = np.object_

import mlflow
import wandb
import hydra
from omegaconf import DictConfig


@hydra.main(config_name='config')
def go(config: DictConfig):
    
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']


    with tempfile.TemporaryDirectory() as tmp_dir:

        pipeline_steps = config['main']['steps']

        if "download" in pipeline_steps:
            _ = mlflow.run(
                "{}/get_data".format(config["main"]["component_repository"]), 
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                }
            )


if __name__ == "__main__":
    go()







