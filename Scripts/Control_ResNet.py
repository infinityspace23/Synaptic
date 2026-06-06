import torch
import torchvision.models as models
from torchvision import datasets, transforms
import umap
import matplotlib.pyplot as plt

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
testSet = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    ##transform = transforms.ToTensor()
)

# Preprocess
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])



activations = []
stacked_activations = []
labels_listing = []

def hook_fn(module, input, output):
    activations.append(output)

model.avgpool.register_forward_hook(hook_fn)

device = torch.device("cpu")
model = model.to(device)

for i in range(1000):
    img,Labl = testSet[i]

    tensor = transform(img).unsqueeze(0)

    input_batch = tensor.to(device)
    
    with torch.no_grad():
        model(input_batch)
        
    stacked_activations.append(torch.flatten(activations[i]))
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
plt.title('ResNet50 Activations — Fashion MNIST (1000 images)')
plt.show()

