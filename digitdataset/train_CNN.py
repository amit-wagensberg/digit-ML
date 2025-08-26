from load import train_loader, test_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns

best_acc = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # input: (1, 90, 140) -> output: (32, 90, 140)
        self.pool = nn.MaxPool2d(2, 2)                            # -> (32, 45, 70)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (64, 45, 70)
        self.pool2 = nn.MaxPool2d(2, 2)                           # -> (64, 22, 35)

        # Flatten size = 64 * 22 * 35
        self.fc1 = nn.Linear(64*22*35, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all but batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

CNN = (CNN().to(device))
print(CNN)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(CNN.parameters(), lr=0.001, momentum=0.9)


def train(train_loader,model,loss_fn,optimizer):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        X = model(images)
        loss = loss_fn(X, labels)
        loss.backward()
        optimizer.step()


def test(test_loader,model,loss_fn,best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            X = model(images)
            test_loss += loss_fn(X, labels).item()
            pred = X.argmax(dim=1)
            correct += (pred == labels).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * (correct / len(test_loader.dataset))
    best_acc = (max(accuracy, best_acc))
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")

epochs = 40
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n")
    train(train_loader, CNN, loss_fn, optimizer)
    test(test_loader, CNN, loss_fn, best_acc)




CNN.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = CNN(data)
        pred = output.argmax(dim=1)  # predicted class
        all_preds.extend(pred.cpu().numpy())    # store as list
        all_targets.extend(target.cpu().numpy())


num_classes = 10
conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)

for t, p in zip(all_targets, all_preds):
    conf_matrix[t, p] += 1

print("Confusion Matrix:\n", conf_matrix)



plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix.numpy(), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print(best_acc)
