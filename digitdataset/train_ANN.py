from load import train_loader, test_loader
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns

best_acc = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(140*90, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model_ANN = ANN().to(device)
print(model_ANN)

def train_ANN(train_loader, model, optimizer, loss_fn):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


def test(test_loader, model, loss_fn,best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * (correct / len(test_loader.dataset))
    best_acc =(max(accuracy, best_acc))
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ANN.parameters(), lr=0.001, momentum=0.9)
epochs = 80
for epoch in range(epochs):
    print(f"Epoch {epoch +1}\n")
    train_ANN(train_loader, model_ANN, optimizer, loss_fn)
    test(test_loader, model_ANN, loss_fn,best_acc)

print("DONE AND GONE")

model_ANN.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model_ANN(data)
        pred = output.argmax(dim=1)  # predicted class
        all_preds.extend(pred.cpu().numpy())    # store as list
        all_targets.extend(target.cpu().numpy())


num_classes = 10  # change to your number of classes
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
