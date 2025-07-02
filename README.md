# STONet: A Neural Operator for Modeling Solute Transport in Micro-Cracked Reservoirs

This repository contains the code and data for the paper "STONet: A neural operator for modeling solute transport in micro-cracked reservoirs" ([arXiv:2412.05576](https://arxiv.org/abs/2412.05576)).

## Introduction

STONet is a deep learning model for predicting solute transport in micro-cracked reservoirs. It is a neural operator, which is a type of neural network that can learn mappings between function spaces. This allows STONet to learn the underlying physics of solute transport and make accurate predictions even for complex and heterogeneous reservoirs.

## Repository Structure

```
├── data/
│   ├── data25_test.csv
│   └── data500_train.csv
├── paper/
│   └── stonet.pdf
├── src/
│   ├── data_models.py
│   ├── losses.py
│   ├── networks.py
│   ├── optimizers.py
│   └── utils.py
├── plot_data.py
├── plot_test.py
├── pred.py
├── train.py
├── LICENSE
└── README.md
```

*   **`data/`**: Contains the training and testing datasets.
*   **`paper/`**: Contains the research paper.
*   **`src/`**: Contains the codes for the STONet model.
    *   `networks.py`: Defines the neural network architectures (e.g., `EnrichedDeepONet`, `STONet`).
    *   `data_models.py`: Handles data loading and preprocessing.
    *   `losses.py`: Defines the loss function used for training.
    *   `optimizers.py`: Contains the optimizer for training the model.
    *   `utils.py`: Contains utility functions.
*   **`plot_data.py`**: Script for visualizing the training data.
*   **`plot_test.py`**: Script for visualizing the test data.
*   **`pred.py`**: Script for making predictions on the test data.
*   **`train.py`**: Script for training the STONet model.

## Getting Started

### Prerequisites

*   Python 3.10+

You can install the required packages using `pip` and the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Training

To train the STONet model, run the `train.py` script:

```bash
python train.py
```

The script will train the model and save the checkpoints and results in the `checkpoints/` and `results/` directories, respectively.

### Prediction

To make predictions on the test data, run the `pred.py` script:

```bash
python pred.py
```

The script will load the latest trained model and generate predictions. The predictions and plots will be saved in the `results/` directory.

### Plotting

To visualize the training and test data, you can use the `plot_data.py` and `plot_test.py` scripts:

```bash
python plot_data.py
python plot_test.py
```

The plots will be saved in the `data/plots/` and `data/test_plots/` directories, respectively.

## Citation

If you use this code or data in your research, please cite the following paper:

```
@article{haghighat2024stonet,
  title={STONet: A neural operator for modeling solute transport in micro-cracked reservoirs},
  author={Haghighat, Ehsan and Adeli, Mohammad Hesan and Mousavi, S Mohammad and Juanes, Ruben},
  journal={arXiv preprint arXiv:2412.05576},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.