import os
import tempfile
import mlflow
import wandb
import hydra
from omegaconf import DictConfig


steps_local = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model"
]

@hydra.main(config_name='config')
def go(config: DictConfig):
    
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']


    with tempfile.TemporaryDirectory() as tmp_dir:

        pipeline_steps = config['main']['steps']

        if "download" in pipeline_steps:
            _ = mlflow.run(
                f{config["main"]["component_repository"]}
            )







