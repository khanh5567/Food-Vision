# Author: Abdella Osman

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

# Define the CNN Model
class Food101CNN(nn.Module):
    def __init__(self, num_classes=101):  # Assuming you have 101 classes
        super(Food101CNN, self).__init__()
        # Load a pre-trained ResNet50 model
        self.model = models.resnet50(pretrained=True)
        # Replace the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

# Function to train the model
def train_model(dataset_path, model_save_path, num_epochs=50, num_passes=10):
    # Define the directory paths
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')

    # Verify the split
    def count_files(directory):
        return sum([len(files) for r, d, files in os.walk(directory)])

    # Print the number of training and testing images
    print(f"Total training images: {count_files(train_dir)}")
    print(f"Total testing images: {count_files(test_dir)}")

    # Define data transformations with more augmentation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    full_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Save class names
    class_names = full_dataset.classes
    classnames_path = os.path.join(model_save_path, 'classnames')
    os.makedirs(classnames_path, exist_ok=True)
    with open(os.path.join(classnames_path, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)

    # Instantiate the model, define the loss function and the optimizer
    model = Food101CNN(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Check if GPU is available and move the model to GPU if it is
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("Using CPU")
    model.to(device)

    # Initialize TensorBoard
    writer = SummaryWriter(f'runs/food_experiment_{dataset_path.split("/")[-1]}')

    # Training Loop with TensorBoard logging
    best_val_accuracy = 0
    for pass_num in range(num_passes):
        print(f"Pass {pass_num + 1}/{num_passes}")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', ncols=100)
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Log batch loss
                writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

            # Validation Loop
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            val_loss = val_loss / len(val_loader)
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
            print(f'Validation Accuracy: {val_accuracy:.2f}%')

            # Step the scheduler
            scheduler.step(val_loss)

            # Save the model weights if validation accuracy improves
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                os.makedirs(model_save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
                print(f"Best model weights saved to {os.path.join(model_save_path, 'best_model.pth')}")

    writer.close()
    return model, device, class_names


#Sample use of the train_model function
if __name__ == '__main__':
    dataset_path = '/home/suprim/Dev/Abdella/pythonProject/finalproject/food-101dataset/data'
    model_save_path = '/home/suprim/Dev/Abdella/pythonProject/finalproject/weight/data'
    train_model(dataset_path, model_save_path, num_epochs=10, num_passes=10)  # Adjust num_epochs and num_passes as needed
