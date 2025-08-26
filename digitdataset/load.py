import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets


transforms = transforms.Compose([transforms.ToTensor(),transforms.Grayscale(num_output_channels=1),transforms.Normalize((0.5,),(0.5,))])

Full_dateset= datasets.ImageFolder(root=r'C:\Users\amitw\OneDrive\Desktop\digit_dataset',transform=transforms)

train_size = int(0.8 * len(Full_dateset))
test_size = len(Full_dateset) - train_size
train_set, test_set = torch.utils.data.random_split(Full_dateset, [train_size, test_size])


train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)



for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)


print("Total images:", len(Full_dateset))
print("Training images:", len(train_set))
print("Testing images:", len(test_set))
print("Classes:", Full_dateset.classes)