from aidose.dataset import HF_HUB_REPO_ID, DATASET_VERSION

import os
import logging
from datasets import load_from_disk

HF_TOKEN = os.environ["HF_TOKEN_DS4DH"]

END_POINT_HF_DATASET_PATH = "/data/sets/CT-DOSING-ERRORS/{}".format(DATASET_VERSION)

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
        private=False,
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
