import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

trainSet = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform = transforms.ToTensor()
)

train_loader = DataLoader(trainSet, 32, True)

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.pool = nn.AdaptiveAvgPool2d ((1,1))
        self.fc = nn.Linear(64, 10)
    def forward(self, x, return_embedding = False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(1)
        if return_embedding == True:
            return x
        else:
            x = self.fc(x)
            return x
        
model = FashionCNN()
    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for images, Labl in train_loader:
        predictions = model(images)
        loss = criterion(predictions, Labl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'fashion_cnn.pth')

