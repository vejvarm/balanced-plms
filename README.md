# balanced-plms

This repository contains scripts for preparing text corpora, scraping SPARQL
data and training a Balanced PLM. It bundles the pre‑training pipeline together
with utilities for dataset collection and analysis.

## Functionality overview

* **Dataset preparation** – scripts under `pretraining/` split raw
  datasets, filter out query language snippets and balance clean/dirty
  samples (`01_prepare_dataset.py`, `00_compare_datasets.py` and
  `03_inject_sparql.py`).
* **Model training** – span‑masked pre‑training with
  `pretraining/02_train.py`, using the custom data collator and
  trainer classes.
* **Data scraping** – a collection of scrapers for Reddit,
  StackOverflow, Wikidata, C4 and others accessed via
  `main_scraping.py`.  The `batch_api` tools in `main_batch_api.py`
  support running OpenAI's batch API for query explanations.
* **Utilities** – `plot_from_trainer_state.py` generates loss and
  perplexity plots from a `trainer_state.json` file.

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

Before training, prepare the dataset using
`pretraining/01_prepare_dataset.py`.  This script splits the raw corpus,
filters it and produces balanced clean and dirty splits.  After the data
is prepared, start training with `pretraining/02_train.py`.  Choose one of
the supported datasets and run:

```bash
python pretraining/02_train.py openwebtext-10k
```

Configuration files under `pretraining/configs/` control dataset paths and other
training options.

To augment a dataset with SPARQL examples, use
`pretraining/03_inject_sparql.py` to mix the explanation corpus into the
prepared blocks before running the training script.

## Scraping entry points

The repository provides convenience wrappers for the various scraping scripts.
Run these commands from the repository root. The wrappers automatically
prepend the current working directory to `PYTHONPATH`, so invoking them from the
repository adds the project itself to Python's module search path. Typical tasks
include scraping SPARQL queries from Reddit, C4, StackOverflow and Wikidata.
To list all available tasks run:

```bash
python main_scraping.py --list
python main_batch_api.py --list
```

Execute a specific task by providing its name followed by any arguments that
the underlying script accepts.  `main_batch_api.py` contains helpers for
submitting jobs to the OpenAI batch API and downloading the results.  For
example:

```bash
python main_scraping.py reddit
python main_batch_api.py download_results
```

## Tests

Run the unit tests with:

```bash
pytest
```
