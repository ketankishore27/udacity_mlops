import argparse
import logging 
import os
import wandb
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type='download_file')
    run.config.update(args)

    logger.info(f"Returning sample {args.sample}")
    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        os.path.join('data', args.sample),
        run
    )

if __name__ == "__main__":

    print("Arg parsing commencing")
    parser = argparse.ArgumentParser(description="Download URL to local destination")
    parser.add_argument("sample", type-str, help="Name of the sample tobe downloaded")
    parser.add_argument("artifact_name", type-str, help="Name of the output artifact")
    parser.add_argument("artifact_type", type-str, help="Output artifact type")
    parser.add_argument("artifact_description", type-str, help="Brief description of the artifact")

    args = parser.parse_args()
    print("Args are parsed")
    print(args)
    go(args)
    



