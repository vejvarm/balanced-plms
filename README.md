# balanced-plms

This repository contains scripts for pre-training and evaluating a Balanced PLM.

## Installation

Install Python dependencies using pip:

```bash
pip install -r requirements.txt
```

Alternatively a conda environment file is provided:

```bash
conda env create -f environment.yml
conda activate balanced-plms
```

## Training

The training pipeline is implemented in `pretraining/02_train.py`. Choose one of
the supported datasets and run:

```bash
python pretraining/02_train.py openwebtext-10k
```

Configuration files under `pretraining/configs/` control dataset paths and other
training options.

## Tests

Run the unit tests with:

```bash
pytest
```
