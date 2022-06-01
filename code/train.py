import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from model import Model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([transforms.Resize((298, 224)),
                                transforms.RandomRotation(degrees=(-90, -90)),
                                transforms.ToTensor(),
                                ])

# Setting the hyperparameters
HYPERPARAMETERS = {
    "epochs": 15,
    "batch_size": 10,
    "learning_rate": 1e-3,
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

model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])

# Training loop
n_total_steps = len(train_loader)
accuracy = 0
train_losses, validation_losses, test_losses = np.array([]), np.array([]), np.array([])

for epoch in range(HYPERPARAMETERS["epochs"]):
    training_loss = 0.0
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
                f'Epoch [{epoch + 1}/{HYPERPARAMETERS["epochs"]}], Step [{i + 1}/{n_total_steps}],',
                f"Loss: {loss.item():.3f}",
            )

    model.eval()
    validation_loss = 0.0
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        validation_loss += loss.item()

    train_losses = np.append(train_losses, training_loss)
    validation_losses = np.append(validation_losses, validation_loss)

print("Finished the training")

# Save the model
PATH = "../model/bean_stage_recogniser_01_06_2022.pth"
torch.save(model.state_dict(), PATH)

# Evaluate the validation loss and accuracy
with torch.no_grad():
    test_loss = 0.0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        results = model(images)
        batch_loss = criterion(results, labels)
        test_loss += batch_loss.item()

        ps = torch.exp(results)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        test_losses = np.append(test_losses, test_loss)
    print(
        f"Test loss: {test_loss / len(test_loader):.3f}\n"
        f"Test accuracy: {accuracy / len(test_loader):.3f}"
    )


# Plot of the training and validation losses
epochs = np.arange(HYPERPARAMETERS["epochs"]) + 1

fig, ax = plt.subplots()

ax.set_xticks(epochs)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss value")
ax.plot(epochs, train_losses/len(train_loader), label="Training loss")
ax.plot(epochs, validation_losses/len(valid_loader), label="Validation loss")
ax.legend(frameon=False)
ax.grid()
plt.show()
