# HCP-DFC-Embedding

## Dependencies

This project requires the following:

### Python Environment
- Python 3.9.0
- cuda 11.8

### Installation

```bash
conda create --name py39 --file requirements.txt -c pytorch -c nvidia
conda install -c nvidia cuda-nvcc=11.8
pip install --no-build-isolation mamba-ssm[causal-conv1d]
pip install ray h5py imblearn optuna

```