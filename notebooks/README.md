# Notebooks

This folder contains Jupyter notebooks for data exploration and experimentation.

## Setup

Create and activate a virtual environment (if you have not already), then install
the additional notebook dependencies:

```bash
pip install -r requirements-dev.txt
```

## Run the EDA notebook

From the repository root:

```bash
jupyter notebook
```

Then open:

- `notebooks/01_eda.ipynb`

The notebook reuses the same synthetic data generation pipeline as the training
code (`training/preprocess.py`) so that the analysis stays consistent with the
modeling code.