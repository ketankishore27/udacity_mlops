import os
import tempfile
import numpy as np
np.object = np.object_

import mlflow
import wandb
import hydra
from omegaconf import DictConfig

default_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]
os.environ['HYDRA_FULL_ERROR'] = '1' ## to delete
os.environ['OC_CAUSE'] = '1' ## to delete
@hydra.main(config_name='config')
def go(config: DictConfig):
    
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']


    with tempfile.TemporaryDirectory() as tmp_dir:

        steps_provided = config["main"]["steps"]
        pipeline_steps = default_steps if steps_provided == 'all' else steps_provided.split(",")
        root_path = hydra.utils.get_original_cwd()
        

        print("Say Hi", pipeline_steps)
        if "download" in pipeline_steps:
            print("Say Hi")
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl_configs"]["data_file"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                }
            )


if __name__ == "__main__":
    go()







