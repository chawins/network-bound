# Bounding Outputs of Neural Network under Various Uncertainty Sets

### Members
- Ryan O’Gorman
- Alicia Yi-Ting Tsai (SID: 3034259700)
- Chawin Sitawarin (SID: 3034213823)
- Ruojie Zeng (SID: 3034193517)

### Files
- `lib/custom_layers.py`: implement first layer of network that turns different uncertainty sets into interval bounds
- `lib/dataset_utils.py`: load and process datasets
- `lib/ibp_layers.py`: implement interval bound propagation as described in the original IBP paper (we reimplement this because the released code is very difficult to modify)
- `lib/mnist_model.py`: contain the neural network used in the experiments
- `train_mnist_custom.py`: the main script for training all of the robust networks
- `train_mnist.py` : a script for training IBP models
- `test_mnist.ipynb` : a Jupyter notebook used for testing models and creating all of the figures

Model weights are not included because it makes the repository too large.
