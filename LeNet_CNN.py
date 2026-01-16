import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------------
# DEFINE NETWORK (LeNet-style)
# -------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))        # (N, 6, 28, 28)
        x = F.max_pool2d(x, 2)           # (N, 6, 14, 14)
        x = F.relu(self.conv2(x))        # (N, 16, 10, 10)
        x = F.max_pool2d(x, 2)           # (N, 16, 5, 5)
        x = torch.flatten(x, 1)          # (N, 400)
        x = F.relu(self.fc1(x))          # (N, 120)
        x = F.relu(self.fc2(x))          # (N, 84)
        x = self.fc3(x)                  # (N, 10)
        return x

# -------------------------------
# CREATE MODEL
# -------------------------------
net = Net()
print(net)

# -------------------------------
# RANDOM INPUT (32x32 image)
# -------------------------------
input_tensor = torch.randn(1, 1, 32, 32)
output = net(input_tensor)
print("Output:", output)

# -------------------------------
# LOSS FUNCTION
# -------------------------------
criterion = nn.MSELoss()

target = torch.randn(1, 10)
loss = criterion(output, target)
print("Loss:", loss.item())

# -------------------------------
# BACKPROPAGATION
# -------------------------------
net.zero_grad()
loss.backward()

print("Gradient of conv1 bias:")
print(net.conv1.bias.grad)

# -------------------------------
# OPTIMIZER (SGD)
# -------------------------------
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.step()

print("Weights updated successfully.")
