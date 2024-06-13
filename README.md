# Food Vision: AI-Powered Food Recognition and Nutritional Analysis

## Overview

This project leverages machine learning and external APIs to create a comprehensive food recognition system. The system consists of two main components: a Convolutional Neural Network (CNN) for classifying food images and an API integration for retrieving nutritional information about the classified food items.

## Objectives

1. **Food Classification**:
    - Train a CNN model using the Food-101 dataset to accurately classify images into one of 101 different food categories.
    - Utilize various data augmentation techniques to improve the model's robustness and accuracy.

2. **Nutritional Information Retrieval**:
    - Integrate with the Edamam API to fetch detailed nutritional information about the classified food items.
    - Provide a user-friendly way to view nutritional details for the predicted food category.

## Components

1. **Food Classification using CNN**:
    - **Dataset**: The project uses the Food-101 dataset, which contains 101,000 images of food items grouped into 101 categories.
    - **Model**: A ResNet-50 architecture is employed, with transfer learning to leverage pre-trained weights and fine-tuning on the Food-101 dataset.
    - **Training**: The model has already been trained on the Food-101 dataset, and the trained model weights are included in this project.

2. **Edamam API Integration**:
    - **API Usage**: After predicting the food category, the system queries the Edamam API to fetch nutritional information such as calories, protein, fat, carbohydrates, vitamins, and minerals.
    - **User Input**: Users need to create an Edamam account and provide their `app_id` and `app_key` to use the API.

## Implementation Details

1. **Training the Model**:
    - **Script**: The `training.py` script handles the training process, including data loading, augmentation, model definition, and training loops.
    - **Hyperparameters**: Key hyperparameters such as learning rate, batch size, and number of epochs are adjustable to fine-tune the training process.

2. **Predicting Food Items**:
    - **Script**: The `prediction.py` script loads the trained model and class names, processes the input image, and outputs the predicted food category.
    - **Inference**: The script uses the trained CNN model to predict the food item from an input image.

3. **Retrieving Nutritional Information**:
    - **Script**: The `api_request.py` script takes the predicted food category and queries the Edamam API to fetch detailed nutritional information.
    - **Output**: The script outputs nutritional details such as calories, protein, fat, carbohydrates, and other nutrients.

## Usage

1. **Training the Model** (Optional):
    - If you wish to retrain the model, download and extract the Food-101 dataset.
    - Run the `training.py` script with appropriate paths to train the CNN model.
    - The trained model weights will be saved for later use.

2. **Combined Classification and Information Retrieval**:
    - Run the `Food_vision.py` script with the path to the input image.
    - The script will output both the predicted food category and its nutritional information.
      
## Installation

To run this project, ensure you have the following libraries installed:
- `torch`
- `torchvision`
- `Pillow`
- `requests`
- `tqdm`
- `tensorboard`

You can install these dependencies using the following command:

```bash
pip install torch torchvision pillow requests tqdm tensorboard
