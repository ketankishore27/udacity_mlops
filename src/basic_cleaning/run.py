#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)
    print(args)
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    dataframe = pd.read_csv(artifact_local_path, index_col="id")
    #print("\n\n\n\n\n", dataframe.dtypes, "min_price", type(args.min_price), "\n\n\n\n\n")
    min_price = int(args.min_price)
    max_price = int(args.max_price)
    idx = dataframe['price'].between(min_price, max_price)
    dataframe = dataframe[idx].copy()
    logger.info("Dataset price outliers removal outside range: %s-%s",
                 min_price, max_price)
    dataframe['last_review'] = pd.to_datetime(dataframe['last_review'])
    logger.info("Dataset last_review data type fix")

    idx = dataframe['longitude'].between(-74.25, -73.50) & dataframe['latitude'].between(40.5, 41.2)
    dataframe = dataframe[idx].copy()

    tmp_artifact_path = os.path.join(args.tmp_directory, args.output_artifact)
    dataframe.to_csv(tmp_artifact_path)
    logger.info("Temporary artifact saved to %s" , tmp_artifact_path)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(tmp_artifact_path)
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("Cleaned dataset uploaded to wandb")

    ######################
    # YOUR CODE HERE     #
    ######################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--tmp_directory", 
        type=str,
        help="Temporary directory name",
        required=True)

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact Name",
        required=True)

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output Artifact Name",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output Artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=str,
        help="Minimum price limit",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=str,
        help="Maximum Price limit",
        required=True
    )

    args = parser.parse_args()

    go(args)
