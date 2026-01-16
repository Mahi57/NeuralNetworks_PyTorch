# ===============================
# MNIST: ANN vs CNN (PyTorch)
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------------
# Device (CPU / GPU)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Load MNIST Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------------------------------
# ANN Model
# -------------------------------
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
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

# -------------------------------
# CNN Model
# -------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------------
# Training Function
# -------------------------------
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# -------------------------------
# Train and Compare Models
# -------------------------------
criterion = nn.CrossEntropyLoss()

# ---- ANN ----
ann = ANN().to(device)
optimizer_ann = optim.Adam(ann.parameters(), lr=0.001)

print("\nTraining ANN...")
train_model(ann, train_loader, criterion, optimizer_ann, epochs=5)
ann_accuracy = evaluate_model(ann, test_loader)
print(f"ANN Test Accuracy: {ann_accuracy:.2f}%")

# ---- CNN ----
cnn = CNN().to(device)
optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.001)

print("\nTraining CNN...")
train_model(cnn, train_loader, criterion, optimizer_cnn, epochs=5)
cnn_accuracy = evaluate_model(cnn, test_loader)
print(f"CNN Test Accuracy: {cnn_accuracy:.2f}%")

# -------------------------------
# Final Comparison
# -------------------------------
print("\n===== Model Comparison =====")
print(f"ANN Accuracy: {ann_accuracy:.2f}%")
print(f"CNN Accuracy: {cnn_accuracy:.2f}%")
