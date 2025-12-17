# CT-medication-errors

This repository contains the materials related to our paper "Establishing a benchmark for the prediction of dosing
errors in interventional clinical trials".

# Using the dataset:

The dataset is hosted under the HuggingFace Hub at the following link (for now private):

`https://huggingface.co/datasets/sssohrab/ct-dosing-errors`

You can use it simply using the below commands:

```python
from aidose.dataset import HF_HUB_REPO_ID

from datasets import load_dataset

ds = load_dataset(
    HF_HUB_REPO_ID,
    split="train"
)

print(ds)
print(ds.features)

```

# Reproducing the dataset:

You should first download an instance of the MedDRA terminology after having created an account under `www.meddra.org`
and agreeing to their terms and conditions. This should then be placed under
`RESOURCES_DIR/MEDDRA/MedDRA_<MEDRA_VERSION>`, where the default version is set to `MedDRA_27_1_English`.

After having cloned this repository, in a separate python environment, you should then install the `aidose` package with
the command:

```bash
pip install -e .
```

This will install all the required dependencies.

You should then specify an environment variable `DATASETS_ROOT`, pointing to the root folder where you want to store the
datasets, such as the `CTGOV`  (>= 9 GBytes, which will be downloaded automatically by running the main script described
next), as well as the final reproduced dataset `CT-DOSING-ERRORS`.

Once these requirements are satisfied, you may then proceed with the dataset creation script:

```bash
python3 aidose/dataset/main.py
``` 

The final dataset will be created under `<DATASETS_ROOT>/<DATASET_NAME>/<DATASET_VERSION/`.

Various global constants are set under `constants.py`, under the relevant packages within this repository.
