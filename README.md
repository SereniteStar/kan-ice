# **KAN for Ice Curve Prediction**

![Python](https://img.shields.io/badge/python-3.9-blue.svg)![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)![KAN](https://img.shields.io/badge/KAN-Kolmogorov_Arnold_Network-green)![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

This repository contains the implementation of the project **"KAN for Ice Curve Prediction"**. We explore the application of **Kolmogorov-Arnold Networks (KAN)** as a powerful alternative to traditional Multi-Layer Perceptrons (MLP) for predicting complex wing icing profiles.

## Key Features
*   **KAN Architecture**: Leverages the unique spline-based structure of KAN to achieve superior fitting capability for icing curves compared to traditional neural networks.
*   **Data Preprocessing**: Implements a robust geometric filtering pipeline to clean noisy or invalid icing datasets by analyzing step-distances and geometric consistency.
*   **Scientific Sampling**: Utilizes **K-Means clustering** to ensure the training set is representative of the entire physical parameter space, optimizing model generalization.
*   **Dimensionality Reduction**: Employs **Principal Component Analysis (PCA)** to efficiently represent high-dimensional coordinate series, significantly reducing computational complexity.

## Methodology
The icing profile is represented as a series of $(x, y)$ coordinates. Our pipeline follows these steps:

- **Geometric Filtering**: Removes physically inconsistent or overlapping curves based on geometric constraints.

- **PCA Dimensionality Reduction**: Projects the icing displacement data into a lower-dimensional latent space.

- **Scientific Data Splitting**: Uses K-Means to select distinct samples for training, ensuring balanced coverage of operational parameters (e.g., AoA, Velocity, LWC).

- **KAN Training**: Trains the network using the LBFGS optimizer to minimize prediction error, achieving high convergence efficiency.

## Evaluation
The model's performance is validated using a comprehensive suite of metrics:
*   **Regression Metrics**: MSE, MAE, and $R^2$ Score.
*   **Geometric Similarity**: Hausdorff distance, Chamfer distance, and IoU (Intersection over Union).

<img src="E:\kan_ice\result\plots\radar_chart.png" alt="radar_chart" style="zoom:60%;" />

## Results
The model demonstrates exceptional performance in icing profile prediction. Prediction curves show high fidelity to the ground truth across diverse operational scenarios, confirming that KAN is a robust and efficient choice for complex geometric regression tasks.

<img src="E:\kan_ice\example.png" alt="example" style="zoom: 20%;" />

## Getting Started

### Prerequisites
To set up the environment, we recommend using [Anaconda](https://www.anaconda.com/). Follow these commands to create and configure the environment:

```bash
# Create a new conda environment
conda create -n kan-ice python=3.9 -y

# Activate the environment
conda activate kan-ice

# Install PyTorch (Ensure CUDA is compatible with your GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install necessary dependencies
pip install pykan
```

### Running the Project
Ensure your data is placed in the `data/` directory.

Run the main training and evaluation script:
```bash
python ice.py
```

Check the `result/` folder for logs, saved models, and prediction results.

## Data Requirements

*(Note: Currently, we are unable to provide the original research dataset due to project constraints. However, the model is designed to be compatible with any custom dataset following the format below. Please ensure your data is organized into the following three CSV files in the `data/` directory:)*

*   **`data.csv`**: Contains physical input parameters (e.g., AoA, Velocity, Altitude, Time, Mvc, T0, LWC).
*   **`containMatrix.csv`**: The ground truth matrix of shape `(N, 1600)`, where `N` is the number of samples. Each row represents a 2D icing curve trajectory.
*   **`diffMatrix.csv`**: The corresponding difference/residual matrix used for PCA training.

*Please ensure the row order in all three files corresponds correctly to the same physical samples.*