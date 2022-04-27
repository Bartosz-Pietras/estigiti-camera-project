import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

from model import Model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.Resize((298, 224)),
                                transforms.ToTensor(),
                                ])

# Adding a random comment just to see if GitHub issues and pull requests work properly.
# Setting the hyperparameters
num_epochs = 20
batch_size = 1
learning_rate = 0.001

# Loading and splitting the data into train/test set
train_data = datasets.ImageFolder('../dataset/', transform=transform)
test_data = datasets.ImageFolder('../dataset/', transform=transform)

test_size = 0.2
num_images = len(train_data)
indices = list(range(num_images))
split = int(np.floor(test_size * num_images))
np.random.shuffle(indices)

train_idx, test_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)
train_loader = torch.utils.data.DataLoader(train_data,
                                           sampler=train_sampler, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data,
                                          sampler=test_sampler, batch_size=batch_size)

model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
accuracy = 0
running_loss, test_loss = 0, 0
train_losses, test_losses = [], []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.3f}')
    train_losses.append(running_loss / len(train_loader))
    running_loss = 0

print('Finished the training')

# Save the model
PATH = '../model/cnn.pth'
torch.save(model.state_dict(), PATH)

# Evaluate the validation loss and accuracy
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        results = model(images)
        batch_loss = criterion(results, labels)
        test_loss += batch_loss.item()

        ps = torch.exp(results)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    test_losses.append(test_loss / len(test_loader))
    print(f"Test loss: {test_loss / len(test_loader):.3f}\n"
          f"Test accuracy: {accuracy / len(test_loader):.3f}")

# Plot of the training and validation losses
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()