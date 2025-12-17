#!/usr/bin/env python3

import os
import logging
from datasets import load_from_disk

HF_HUB_REPO_ID = "sssohrab/ct-dosing-errors"
HF_TOKEN = os.environ["HF_TOKEN"]

END_POINT_HF_DATASET_PATH = "/data/sets/CT-DOSING-ERRORS/0.2.2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Loading dataset from disk")
    dataset = load_from_disk(END_POINT_HF_DATASET_PATH)

    logger.info("Pushing dataset to Hugging Face Hub (clean repo)")
    dataset.push_to_hub(
        HF_HUB_REPO_ID,
        token=HF_TOKEN,
        private=True,
        embed_external_files=True,
        max_shard_size="1GB",
    )

    logger.info("Push completed successfully")
    logger.info(
        "Repo: https://huggingface.co/datasets/%s",
        HF_HUB_REPO_ID,
    )


if __name__ == "__main__":
    main()
