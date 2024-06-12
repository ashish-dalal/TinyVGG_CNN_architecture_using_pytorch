# TinyVGG CNN Architecture using PyTorch

This repository contains the implementation of the TinyVGG Convolutional Neural Network (CNN) architecture using PyTorch. TinyVGG is a simplified version of the VGGNet model, designed for educational purposes and small-scale image classification tasks. The model is trained on a custom dataset of pizza, steak, and sushi images, demonstrating the effectiveness of deep learning techniques on a small dataset. The project includes data preprocessing scripts, model training code, and evaluation metrics to help users understand and implement CNNs for image classification.


## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

TinyVGG is a lightweight CNN model inspired by VGGNet. It is designed for educational purposes and small-scale image classification tasks. This repository provides the code for data preprocessing, model training, and evaluation.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ashish-dalal/TinyVGG_CNN_architecture_using_pytorch.git
    cd TinyVGG_CNN_architecture_using_pytorch
    ```
2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset

The dataset used for training and evaluation consists of images of pizza, steak, and sushi. The data is organized in the `data/` directory as follows:
data/

    ├── pizza_steak_sushi/
        
        ├── train/
        
        ├── test/


## Usage

1. **Data Setup**: Prepare the dataset by running the data setup script.
    ```sh
    python data_setup.py
    ```

2. **Model Training**: Train the TinyVGG model using the training script.
    ```sh
    python train.py
    ```

## Training

The training process involves the following steps:
1. Load and preprocess the dataset.
2. Define the TinyVGG model architecture in `model_builder.py`.
3. Train the model using the training script (`train.py`) which includes:
    - Model initialization
    - Loss function and optimizer setup
    - Training loop alongwith testing loop 

## Results

The results of the training process, including accuracy and loss metrics, are saved in the `results/` directory. Model checkpoints are also saved for future use.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
