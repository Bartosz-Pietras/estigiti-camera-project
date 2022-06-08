import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils import evaluate_model, select_proper_device
from model import Model

# Device configuration
device = select_proper_device()
print(f"Chosen device is: {device}")

transform = transforms.Compose([transforms.Resize((298, 224)),
                                transforms.RandomRotation(degrees=(-90, -90)),
                                transforms.ToTensor(),
                                ])

# Setting the hyperparameters
HYPERPARAMETERS = {
    "epochs": 35,
    "batch_size": 32,
    "learning_rate": 1e-2,
    "train_size": 0.7,
    "valid_size": 0.2,
}

# Loading and splitting the data into train/test set
bean_dataset = datasets.ImageFolder("../dataset/", transform=transform)

num_images = len(bean_dataset)
train_count = int(HYPERPARAMETERS["train_size"] * num_images)
valid_count = int(HYPERPARAMETERS["valid_size"] * num_images)
test_count = num_images - train_count - valid_count

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    bean_dataset, (train_count, valid_count, test_count)
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False
)

train_size = len(train_loader)
valid_size = len(valid_loader)
validation_loss = 0.0
training_loss = 0.0

model = Model().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])
criterion = nn.CrossEntropyLoss()


train_losses, validation_losses = np.array([]), np.array([])

# Training loop
for epoch in range(HYPERPARAMETERS["epochs"]):

    lowest_validation_loss = 100000
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        output = model(images)
        loss = criterion(output, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{HYPERPARAMETERS["epochs"]}], Step [{i + 1}/{len(train_loader)}],',
                f"Loss: {loss.item():.3f}",
            )

    model.eval()

    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        validation_loss += loss.item()

    if validation_loss < lowest_validation_loss:
        # Save the model
        PATH = "../model/bean_stage_recogniser_01_06_2022.pth"
        torch.save(model.state_dict(), PATH)
        lowest_validation_loss = validation_loss

    train_losses = np.append(train_losses, training_loss)
    validation_losses = np.append(validation_losses, validation_loss)

print("Finished the training")

# Evaluate the validation loss and accuracy
epochs = np.arange(HYPERPARAMETERS["epochs"]) + 1

evaluate_model(
    model, test_loader, epochs, train_size, valid_size, train_losses, validation_losses
)
