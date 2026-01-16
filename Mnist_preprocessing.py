# ===============================
# MNIST PREPROCESSING + ANN (PyTorch)
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

# -------------------------------
# DEVICE CONFIGURATION
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# PREPROCESSING PIPELINE
# -------------------------------
# 1) Convert image to Tensor (0–255 → 0–1)
# 2) Normalize (mean & std of MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -------------------------------
# LOAD MNIST DATASET
# -------------------------------
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# -------------------------------
# DATALOADERS (LARGE DATA FRIENDLY)
# -------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# -------------------------------
# VISUALIZE ONE PREPROCESSED IMAGE
# -------------------------------
images, labels = next(iter(train_loader))
img = images[0].squeeze().numpy()
plt.imshow(img, cmap="gray")
plt.title(f"Label: {labels[0].item()}")
plt.axis("off")
plt.show()

# -------------------------------
# ANN MODEL DEFINITION
# -------------------------------
class MNIST_ANN(nn.Module):
    def __init__(self):
        super(MNIST_ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MNIST_ANN().to(device)
print(model)

# -------------------------------
# LOSS FUNCTION & OPTIMIZER
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# TRAINING LOOP
# -------------------------------
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# -------------------------------
# EVALUATION
# -------------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# PLOT TRAINING LOSS
# -------------------------------
plt.figure()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# -------------------------------
# PREDICT SINGLE IMAGE
# -------------------------------
sample_img, sample_label = test_dataset[5]
sample_img = sample_img.unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(sample_img)
    predicted_label = torch.argmax(prediction, dim=1).item()

plt.imshow(sample_img.cpu().squeeze(), cmap="gray")
plt.title(f"True: {sample_label}, Predicted: {predicted_label}")
plt.axis("off")
plt.show()
