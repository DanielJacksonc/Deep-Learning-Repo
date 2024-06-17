# Image Classification with Neural Networks

A deep learning project for classifying images into happy and sad categories using convolutional neural networks (CNNs).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates the use of convolutional neural networks (CNNs) for image classification. The model is trained to classify images into two categories: happy and sad. This is a beginner-friendly project designed to help you understand the basics of deep learning and image classification.

## Features

- Classifies images into happy and sad categories
- Utilizes a convolutional neural network (CNN)
- Implements data augmentation to improve model performance
- Provides visualizations of training progress and model predictions
- logs every activity

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/DanielJacksonc/DEEP-LEARNING
    cd image-classification
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```
3. install gpu:
    ```sh
    since our computer has cpu, and its slower than GPU, we need to install GPU in pur tensorflow.
    make sure your python version is 3.9.5, or it will not work since it deprecated.
    and install tensor-gpu 
      # On Windows use `venv\Scripts\activate`
    ```

4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset and place the images in the `data/` directory.
   ```sh
   clean up the dodgy images(images with wrong format, sizes,and not compatible)
   ```

2. Train the model:
    ```sh
    split into training, testing and validation
    python train.py --data_dir data/
    ```

3. Evaluate the model:
    ```sh
    python evaluate.py --model_dir models/
    ```

4. Make predictions on new images:
    ```sh
    python predict.py --image_path path/to/image.jpg
    ```

## Dataset

The dataset consists of images categorized into two classes: happy and sad. You can download the dataset from [this link](https://www.google.com/search?q=very+ugly+car&sca_esv=064fbd3969a095d1&udm=2&biw=1536&bih=695&sxsrf=ADLYWILhxwjva1Lvbh1BKs8a9x3yBWQYeA%3A1718639580545&ei=3FtwZsf2INHIwN4Pi_-UmAM&ved=0ahUKEwjH1Met_-KGAxVRJNAFHYs_BTMQ4dUDCBA&uact=5&oq=very+ugly+car&gs_lp=Egxnd3Mtd2l6LXNlcnAiDXZlcnkgdWdseSBjYXIyBRAAGIAEMgUQABiABDIHEAAYgAQYGEiEJ1DgCViTI3ADeACQAQCYAZIBoAGVB6oBBDEyLjG4AQPIAQD4AQGYAg2gArgHwgIEECMYJ8ICCBAAGIAEGLEDwgIOEAAYgAQYsQMYgwEYigXCAgoQABiABBhDGIoFwgILEAAYgAQYsQMYgwHCAgYQABgFGB6YAwDiAwUSATEgQIgGAZIHBDEyLjGgB99D&sclient=gws-wiz-serp#imgrc=9soo3VKgwTZMgM&imgdii=uJlFdJn4GAl9HM). Ensure that the images are placed in the `data/` directory with the following structure:



## Model Architecture

The model is built using a convolutional neural network (CNN) with the following layers:

- Convolutional layer with 16 filters and a 3x3 kernel shape and stide of 1, passed through relu activation and then shows how our input looks like(256,256,3)
- Max pooling layer with a 2x2 pool size
- Convolutional layer with 32 filters and a 3x3 kernel
- ReLU activation
- Max pooling layer with a 2x2 pool size
- Fully connected layer with 128 units
- ReLU activation
- then we Flatten our output layer
- Output layer with Relu and sigmoid activation

## Results

### Training and Validation Accuracy

![LOSS][alt text](image.png)

### Loss

(![ACCURACY]!(image-1.png))

### Sample Predictions

| Image | Prediction | Actual |
|-------|------------|--------|
| ![Beautiful](![alt text](image-2.png)) | Beautiful  | Beautiful |
| ![saUglyd](![alt text](image-3.png)) | Ugly | Ugly |


## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -)




