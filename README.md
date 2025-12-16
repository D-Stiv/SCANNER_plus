# SCANNER+
Implementation of the paper "SCANNER+: Neighborhood-based self-enrichment approach for traffic speed prediction", accepted for publication in the ACM Transactions on Spatial Algorithms and Systems (TSAS) Special Issue on Urban Mobility.

## Summary
In this repository, you can find the code to train and evalute SCANNER+, a novel neighborhood-based self-enrichment approach for traffic speed prediction. SCANNER+ learns effective node representations in dynamic road traffic settings.

## Setup
- Create a new virtual environment, and activate.
```bash
virtualenv venv
source venv/bin/activate
```

- Install Pytorch and Pytorch Geometric
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install torch-geometric
pip install -r requirements.txt
```

If you find this repository useful for your research, please consider citing the following paper:
```bash
@article{gounoueTSAS25,
  author  = {Gounoue Guiffo, Steve and Markwald, Marco and Yu, Ran and Demidova, Elena},
  title   = {{SCANNER+}: Neighborhood-based self-enrichment approach for traffic speed prediction},
  journal = {ACM Transactions on Spatial Algorithms and Systems},
  year    = {2025},
}
```

<!-- 
## Data
metr-la and pems-bay datasets from the paper:
```
@inproceedings{DBLP:conf/iclr/LiYS018,
  author       = {Yaguang Li and
                  Rose Yu and
                  Cyrus Shahabi and
                  Yan Liu},
  title        = {Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
                  Forecasting},
  booktitle    = {Proceedings of the 6th International Conference on Learning Representations, {ICLR} 2018},
 publisher    = {OpenReview.net},
  year         = {2018},
}
``` 
-->

## Prerequisites
1. Compute the temporal correlation matrices **B** and save it in .pkl format. **B** has shape ```L X N X N```, where ```L``` is the number of lags and ```N``` the number of nodes.
2. Save the spatial distance matrix **A** in .pkl format. A has shape ```N X N```.
3. Save the dataset **W** in .h5 format. **W** has shape ```T_max X N```, where ```T_max``` is the total number of time steps in the data. The index of **W** should be castable in datetime type.
4. Substitute in the code the path corresponding to the different locations.
    * Temporal correlation matrices **B**: correlation.py
    * Spatial distance matrix **A**: correlation.py
    * Dataset: loader.py
    * Model checkpoints: main.py

## Run the Code
Configuration parameters are in the file ```config.py```. Example with metr-la for maximum 200 epochs.
```bash
python3 main.py --dataset_name metr-la --epochs 200
```