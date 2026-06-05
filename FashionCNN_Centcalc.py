# Mostly same as the initial FashionCNN_Test.py, 
# but this has some code for calculation of centroid for each label

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import umap
import matplotlib.pyplot as plt

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.pool = nn.AdaptiveAvgPool2d ((1,1))
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(1)
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


stacked_activations = {'conv1':[], 'conv2':[], 'avgpool':[]}
labels_listing = []
all_hooks = {'conv1':[], 'conv2':[], 'avgpool':[]}

def get_layer(layer):
    def hook_fn(module, input, output):
        all_hooks[layer].append(output)
    return hook_fn

model.conv1.register_forward_hook(get_layer('conv1'))
model.conv2.register_forward_hook(get_layer('conv2'))
model.pool.register_forward_hook(get_layer('avgpool'))

device = torch.device("cpu")
model = model.to(device)

for i in range(1000):
    img,Labl = testSet[i]

    input_batch = img.to(device)
    
    with torch.no_grad():
        model(input_batch.unsqueeze(0))

    stacked_activations['conv1'].append(all_hooks['conv1'][i].flatten(0))
    stacked_activations['conv2'].append(all_hooks['conv2'][i].flatten(0))
    stacked_activations['avgpool'].append(all_hooks['avgpool'][i].flatten(0))

    labels_listing.append(Labl) # Will be used for boolean indexing later on for centroid calculation


class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

reducer = umap.UMAP(n_components=2, random_state=42)

# Following code is only for plotting UMAP

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 8))

# for i in all_hooks:
#     matrix = torch.stack(stacked_activations[i]).detach().numpy()
#     embedding = reducer.fit_transform(matrix)
    
#     for class_idx in range(10):
#         mask = [l == class_idx for l in labels_listing]
#         if i == 'conv1':
#             ax1.scatter(embedding[mask, 0], embedding[mask, 1], 
#                     label=class_names[class_idx], s=10)
#             ax1.set_title('Conv1 activations (1000 imgs)')
#             ax1.legend()
#         elif i == 'conv2':
#             ax2.scatter(embedding[mask, 0], embedding[mask, 1], 
#                     label=class_names[class_idx], s=10)
#             ax2.set_title('Conv2 activations (1000 imgs)')
#             ax2.legend()
#         elif i == 'avgpool':
#             ax3.scatter(embedding[mask, 0], embedding[mask, 1], 
#                     label=class_names[class_idx], s=10)
#             ax3.set_title('Avgpool activations (1000 imgs)')
#             ax3.legend()

# plt.show()

# Centroid calculation

centroid = {0: [], 1: [], 2:[], 3:[], 4:[], 5: [], 6: [], 7: [], 8: [], 9: []}

matrix = torch.stack(stacked_activations['avgpool']).detach().numpy()

for class_idx in range(10):
    mask = [l == class_idx for l in labels_listing]
    centroid[class_idx] = matrix[mask].mean(axis=0)

final_comp = np.stack(list(centroid.values()), axis=0) # Used in sentence_tr.py to compare with all-minilm-l6-v2

