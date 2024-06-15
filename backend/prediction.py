# Author: Abdella Osman

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Define the CNN Model
class Food101CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Food101CNN, self).__init__()
        # Load a pre-trained ResNet50 model
        self.model = models.resnet50(pretrained=True)
        # Replace the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)


# Function to load a model
def load_model(model_path, class_names_path):
    # Check if the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Check if the class names file exists
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")

    # Load the class names from the JSON file
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    num_classes = len(class_names)
    # Initialize the model
    model = Food101CNN(num_classes=num_classes)
    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Set the model to evaluation mode
    model.eval()

    return model, class_names


# Predefined paths for the model and class names
model_path = "weights/data/best_model.pth"
class_names_path = "weights/data/classnames/class_names.json"

# Load the model and class names once
model, class_names = load_model(model_path, class_names_path)


# Function to predict an image
def predict_image(image):
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize the image
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])

    # Open and preprocess the image
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)  # Move image to the device

    model.to(device)  # Move model to the device

    # Predict the class of the image
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = class_names[predicted.item()]
        formatted_class = predicted_class.replace('_', ' ').title()  # Format the class name
    #print("This is an image of a", formatted_class)
    return formatted_class
