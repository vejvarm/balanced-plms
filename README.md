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

## Scraping entry points

The repository provides convenience wrappers for the various scraping scripts.
Run these commands from the repository root. The wrappers automatically
prepend the current working directory to `PYTHONPATH`, so invoking them from the
repository adds the project itself to Python's module search path. To list the
available tasks run:

```bash
python main_scraping.py --list
python main_batch_api.py --list
```

Execute a specific task by providing its name followed by any arguments that
the underlying script accepts. For example:

```bash
python main_scraping.py reddit
python main_batch_api.py download_results
```

## Tests

Run the unit tests with:

```bash
pytest
```
