# SCANNER_plus

## Prerequisites:
1) Compute the temporal correlation B and save it in .pkl format. B has shape L X N X N, where L is the number of lags and N the number of nodes
2) Save the spatial similarity matrix A in .pkl format. A has shape M X N
3) Save the dataset W in .h5 format. W has shape T_max X N, where T_max is the total number of time steps in the data. The index of W should be castable in datetime type.
4) Substitute in the code the path corresponding to the different locations.

## Run the model
`python3 run.py --dataset_name {name of the dataset}`