import os
import tempfile
import numpy as np
np.object = np.object_

import mlflow
import json
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

        if "download" in pipeline_steps:
            _ = mlflow.run(
                f"{config['main']['component_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl_configs"]["data_file"],
                    "artifact_name": "sample1.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                }
            )

        if "basic_cleaning" in pipeline_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "basic_cleaning"),
                "main",
                parameters={
                    "tmp_directory": tmp_dir,
                    "input_artifact": "sample1.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl_configs']['min_price'],
                    "max_price": config['etl_configs']['max_price']
                }
            )

        if "data_check" in pipeline_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config['data_checks']['kl_threshold'],
                    "min_price": config['etl_configs']['min_price'],
                    "max_price": config['etl_configs']['max_price']
                },
            )   

        if "data_split" in pipeline_steps:
            _ = mlflow.run(
                os.path.join(root_path, "components", "train_val_test_split"),
                "main",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                },
            )

        if "train_random_forest" in pipeline_steps:

            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as file_p:
                json.dump(
                    dict(
                        config["modeling"]["random_forest"].items()),
                    file_p) 

            _ = mlflow.run(
                os.path.join(
                    root_path,
                    "src",
                    "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": config["modeling"]["output_artifact"]},
            )

        if "test_regression_model" in pipeline_steps:
            _ = mlflow.run(
                os.path.join(root_path, "components", "test_regression_model"),
                "main",
                parameters={
                    "mlflow_model": config["modeling"]["output_artifact"] + ":prod",
                    "test_dataset": "test_data.csv:latest"
                }
            )

if __name__ == "__main__":
    go()







