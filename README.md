# SCANNER+

Implementation of the paper **“SCANNER+: Neighborhood-based self-enrichment approach for traffic speed prediction”**, accepted for publication in the *ACM Transactions on Spatial Algorithms and Systems (TSAS)*.

## Summary

This repository contains the code to train and evaluate **SCANNER+**, a novel neighborhood-based self-enrichment approach for traffic speed prediction. SCANNER+ learns effective node representations in dynamic road traffic settings by leveraging spatio-temporal correlations.

This work extends **SCANNER**, which utilizes correlation-based pattern detection and a self-enrichment mechanism:

* Paper: [https://doi.org/10.1145/3589132.3625653](https://doi.org/10.1145/3589132.3625653)
* Code repository: [https://github.com/D-Stiv/SCANNER](https://github.com/D-Stiv/SCANNER)

The archived dataset used in this work is available via **bonndata**:

* Dataset DOI: [https://doi.org/10.60507/FK2/DRYP80](https://doi.org/10.60507/FK2/DRYP80)

## Code Repository

The official GitHub repository for this project is:

* **[https://github.com/D-Stiv/SCANNER_plus](https://github.com/D-Stiv/SCANNER_plus)**

## Tool Version

This repository corresponds to:

* **SCANNER+ v1.0.0** (initial research release accompanying the TSAS publication)

Future updates or experimental extensions should be versioned separately.


## System Requirements

The model was developed and tested under the following conditions:

* **Operating System**: Any 64-bit OS supporting Python and PyTorch
  (e.g., Linux, macOS, Windows)
* **Python**: 3.8.30
* **PyTorch**: 1.9.0+cu111
* **NumPy**: 1.23.5
* **Architecture**: 64-bit machine
* **Containerization**:
  No container (e.g., Docker) is required. The code runs in a standard Python virtual environment.

GPU support is optional but recommended for faster training.

## Setup

Create and activate a virtual environment:

```bash
virtualenv venv
source venv/bin/activate
```

Install PyTorch, PyTorch Geometric, and remaining dependencies:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install torch-geometric
pip install -r requirements.txt
```

## Prerequisites

1. Compute the temporal correlation matrices **B** and save them in `.pkl` format.

   * Shape: `L × N × N`
   * `L`: number of temporal lags
   * `N`: number of nodes

2. Save the spatial distance matrix **A** in `.pkl` format.

   * Shape: `N × N`

3. Save the dataset **W** in `.h5` format.

   * Shape: `T_max × N`
   * The index of **W** must be castable to `datetime`.

4. Update file paths in the source code:

   * Temporal correlation matrices **B**: `correlation.py`
   * Spatial distance matrix **A**: `correlation.py`
   * Dataset **W**: `loader.py`
   * Model checkpoints: `main.py`

## Run the Code

Configuration parameters are defined in `config.py`.

Example command using the *metr-la* dataset for 200 epochs:

```bash
python3 main.py --dataset_name metr-la --epochs 200
```

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{gounoueTSAS26,
  author  = {Gounoue Guiffo, Steve and Markwald, Marco and Yu, Ran and Demidova, Elena},
  title   = {{SCANNER+}: Neighborhood-based self-enrichment approach for traffic speed prediction},
  journal = {ACM Transactions on Spatial Algorithms and Systems},
  year    = {2026},
}
```

## License
This repository contains code from other sources under ``MIT License``

We re-implement (in ``stnorm.py``) the spatio-temporal normalization from the the paper
"ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting"
(https://doi.org/10.1145/3447548.3467330)

We extend Graph-Wavenet (Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019), inserting the spatio-temporal normalization (``gwnet`` in ``models.py``).

The following files are derivative works from Graph-Wavenet
- loader.py: incorporated spatio-temporal correlation in data loading
- metrics.py: incorporated spatio-temporal correlation for model evaluation
- models.py: we integrated spatial and temporal normalization into ``gwnet`` and inserted correlation for neighborhood enrichment



Modifications are indicated in individual file header.
