from aidose.dataset import HF_HUB_REPO_ID, DATASET_VERSION

import os
import re
import logging
from datasets import load_from_disk
from huggingface_hub import DatasetCard

HF_TOKEN = os.environ["HF_TOKEN_DS4DH"]
END_POINT_HF_DATASET_PATH = f"/data/sets/CT-DOSING-ERRORS/{DATASET_VERSION}"

LICENSE_ID = "cc-by-4.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"(?s)\A---\n(.*?)\n---\n")


def _upsert_front_matter(readme_text: str, updates: dict[str, str]) -> str:
    """
    Update (or create) YAML front-matter keys without touching the rest of README.
    Assumes simple 'key: value' lines for the keys we manage.
    """
    m = _FRONT_MATTER_RE.match(readme_text)

    if m:
        yaml_block = m.group(1)
        body = readme_text[m.end():]
        lines = yaml_block.splitlines()
    else:
        lines = []
        body = readme_text

    # Build an index of existing keys -> line index
    key_to_idx: dict[str, int] = {}
    for i, line in enumerate(lines):
        # Only handle simple "key: ..." (ignore nested YAML)
        if ":" in line and not line.lstrip().startswith("#"):
            k = line.split(":", 1)[0].strip()
            if k:
                key_to_idx[k] = i

    def fmt_value(v: str) -> str:
        # Quote only when needed; version typically benefits from quotes.
        if any(ch in v for ch in [":", "#", "{", "}", "[", "]", ",", "\"", "'"]) or v.strip() != v:
            return f"\"{v}\""
        return v

    for k, v in updates.items():
        new_line = f"{k}: {fmt_value(v)}"
        if k in key_to_idx:
            lines[key_to_idx[k]] = new_line
        else:
            lines.append(new_line)

    new_yaml = "\n".join(lines).strip()
    new_readme = f"---\n{new_yaml}\n---\n{body.lstrip()}"
    return new_readme


def _patch_existing_dataset_card(repo_id: str, token: str) -> None:
    # Load existing README.md from the Hub (preserves your schema sections)
    card = DatasetCard.load(repo_id, repo_type="dataset", token=token)
    old = card.content

    patched = _upsert_front_matter(
        old,
        updates={
            "license": LICENSE_ID,
            # "version" isn't a standard hub field; keep it as custom metadata.
            # You can rename this key if you already use something else.
            "dataset_version": DATASET_VERSION,
        },
    )

    if patched == old:
        logger.info("README.md already contains the desired metadata; no change needed.")
        return

    DatasetCard(patched).push_to_hub(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Add/update license/version metadata (v{DATASET_VERSION})",
    )


def main() -> None:
    logger.info("Loading dataset from disk")
    dataset = load_from_disk(END_POINT_HF_DATASET_PATH)

    logger.info("Pushing dataset to Hugging Face Hub")
    dataset.push_to_hub(
        HF_HUB_REPO_ID,
        token=HF_TOKEN,
        private=False,
        embed_external_files=True,
        max_shard_size="1GB",
        commit_message=f"Upload dataset v{DATASET_VERSION}",
    )

    logger.info("Patching existing dataset card (preserve schema; add license/version)")
    _patch_existing_dataset_card(HF_HUB_REPO_ID, HF_TOKEN)

    logger.info("Done")
    logger.info("Repo: https://huggingface.co/datasets/%s", HF_HUB_REPO_ID)


if __name__ == "__main__":
    main()
