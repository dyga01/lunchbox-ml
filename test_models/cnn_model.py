"""Experimenting with Modern CNN Architectures in PyTorch Lab."""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# Determine if CUDA (GPU) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a uniform size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load the FGVCAircraft dataset for training and validation
train_dataset = datasets.FGVCAircraft(root='./data', split='train', transform=transform, download=True)
val_dataset = datasets.FGVCAircraft(root='./data', split='val', transform=transform, download=True)

# Create data loaders to handle batching and shuffling
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

# Load pre-trained DenseNet and EfficientNet models
densenet_model = models.densenet121(pretrained=True)
efficientnet_model = models.efficientnet_b0(pretrained=True)

# Modify the final classification layer for FGVC Aircraft (100 classes)
num_classes = 100
densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, num_classes)
efficientnet_model.classifier = nn.Linear(efficientnet_model.classifier[1].in_features, num_classes)

# Move the models to the GPU if available
densenet_model.to(device)
efficientnet_model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
densenet_optimizer = torch.optim.Adam(densenet_model.parameters(), lr=0.001)
efficientnet_optimizer = torch.optim.Adam(efficientnet_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Number of training epochs

def train_model(model, optimizer, train_loader, val_loader, num_epochs):
    """
    Train the given model using the specified optimizer and data loaders.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        None
    """
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for images, labels in train_loader:
            # Move data to the GPU if available
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the training loss for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            for images, labels in val_loader:
                # Move data to the GPU if available
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print the validation accuracy
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time for the model: {elapsed_time:.2f} seconds")

# Train DenseNet model
print("Training DenseNet model...")
train_model(densenet_model, densenet_optimizer, train_loader, val_loader, num_epochs)

# Train EfficientNet model
print("Training EfficientNet model...")
train_model(efficientnet_model, efficientnet_optimizer, train_loader, val_loader, num_epochs)

# Save the trained models
torch.save(densenet_model.state_dict(), "densenet121_fgvc_aircraft.pth")
torch.save(efficientnet_model.state_dict(), "efficientnet_b0_fgvc_aircraft.pth")

# Visualize the first layer's learned filters for DenseNet
first_layer_weights = densenet_model.features.conv0.weight.data.cpu()
num_filters_to_display = 16
plt.figure(figsize=(8, 8))
for i in range(num_filters_to_display):
    plt.subplot(4, 4, i + 1)
    plt.imshow(first_layer_weights[i][0], cmap='gray')  # Display the first channel in grayscale
    plt.axis('off')
plt.suptitle('DenseNet First Layer Convolutional Filters')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
plt.show()

# Visualize the first layer's learned filters for EfficientNet
first_layer_weights = efficientnet_model.features[0][0].weight.data.cpu()
plt.figure(figsize=(8, 8))
for i in range(num_filters_to_display):
    plt.subplot(4, 4, i + 1)
    plt.imshow(first_layer_weights[i][0], cmap='gray')  # Display the first channel in grayscale
    plt.axis('off')
plt.suptitle('EfficientNet First Layer Convolutional Filters')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
plt.show()

# Feature map visualization
def visualize_feature_maps(model, img, label):
    """
    Visualize the feature maps of the first convolutional layer of the given model.

    Args:
        model (torch.nn.Module): The model to visualize feature maps for.
        img (torch.Tensor): The input image tensor.
        label (int): The true label of the input image.

    Returns:
        None
    """
    # Get the first convolutional layer
    layer = model.features.conv0 if isinstance(model, models.DenseNet) else model.features[0][0]
    
    # Forward, saving the output
    img_tensor = img.unsqueeze(0).to(device)
    output = layer(img_tensor)

    # Get prediction
    with torch.no_grad():
        model.eval()
        prediction = model(img_tensor)
        predicted_class = torch.argmax(prediction).item()

    # Plot the feature maps
    n_feats = output.shape[1]
    fig = plt.figure(figsize=(12, 12))
    
    # Plot original image
    plt.subplot(4, 4, 1)
    img_np = img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    plt.title(f"Original, Predicted: {predicted_class}, Label: {label}")
    plt.axis('off')
    
    # Plot feature maps
    for i in range(min(n_feats, 15)):
        plt.subplot(4, 4, i + 2)
        plt.imshow(output[0, i].data.cpu().numpy(), cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Get a sample image from the validation set
dataiter = iter(val_loader)
images, labels = next(dataiter)
img = images[0]
label = labels[0]

# Visualize feature maps for DenseNet
print("Visualizing feature maps for DenseNet...")
visualize_feature_maps(densenet_model, img, label)

# Visualize feature maps for EfficientNet
print("Visualizing feature maps for EfficientNet...")
visualize_feature_maps(efficientnet_model, img, label)
