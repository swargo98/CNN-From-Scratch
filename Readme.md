# CNN From Scratch for Bangla Character Recognition Challenge

This repository contains a Convolutional Neural Network (CNN) implemented from scratch including the forward and backward propagation steps using numpy. This project focuses on implementing a neural network model to recognize Bangla characters. The task was approached using preprocessing techniques and the LeNet architecture to achieve character classification.

## Project Overview

- **Objective:** Develop a system to recognize Bangla characters effectively.
- **Model Used:** LeNet
- **Key Features:**
  - Preprocessing steps including grayscale conversion, color inversion, and resizing.
  - Experimentation with different learning rates for optimization.
  - Results analyzed for training loss, validation loss, validation accuracy, and macro F1 score.

## Project Setup

- **CPU:** AMD Ryzen 5 3600
- **RAM:** 24 GB
- **Editor:** PyCharm 2022.1.1

## Preprocessing

Before training, the input data underwent the following steps:
1. Conversion to grayscale.
2. Color inversion for better contrast.
3. Resizing images to a consistent input size for the model.