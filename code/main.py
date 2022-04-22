import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

from model import Model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                ])

# Adding a random comment just to see if GitHub issues and pull requests work properly.
# Setting the hyperparameters
num_epochs = 5
batch_size = 16
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
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')

# Save the model
PATH = '../model/cnn.pth'
torch.save(model.state_dict(), PATH)

# Visualize the loss and results
