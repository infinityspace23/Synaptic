import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt

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
model.load_state_dict(torch.load('fashion_cnn.pth'))
model.eval()
testSet = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform = transforms.ToTensor()
)

# Preprocess
#transform = transforms.Compose([
    #transforms.Lambda(lambda img: img.convert("RGB")),
    #transforms.Resize((224,224)),
    ##transforms.CenterCrop(224),
    #transforms.ToTensor(),
#])


activations = []
stacked_activations = []
labels_listing = []


device = torch.device("cpu")
model = model.to(device)

for i in range(1000):
    img,Labl = testSet[i]

    input_batch = img.to(device)
    
    with torch.no_grad():
        stacked_activations.append(model(input_batch, return_embedding = True).squeeze())

    labels_listing.append(Labl)

reducer = umap.UMAP(n_components=2, random_state=42)
matrix = torch.stack(stacked_activations).detach().numpy()
embedding = reducer.fit_transform(matrix)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

plt.figure(figsize=(10, 8))
for class_idx in range(10):
    mask = [l == class_idx for l in labels_listing]
    plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                label=class_names[class_idx], s=10)

plt.legend(markerscale=2)
plt.title('Fashion CNN activations— Fashion MNIST (1000 images)')
plt.show()