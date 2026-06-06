# RSA Heatmap: FashionCNN vs ResNet50
# Computes pairwise Euclidean/Cosine distances for each layer pair across both models,
# then correlates them using Spearman's r to produce a 3x6 RSA matrix.

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Control_ResNet import stacked_activationsR, all_hooksR
from FashionCNN_Centcalc import stacked_activations, all_hooks
from scipy.spatial.distance import cdist 
from scipy import stats

ResNet_Data = {'conv1':[], 'layer1':[],'layer2':[],'layer3':[],'layer4':[],'avgpool':[]}
ResNet_cdist = []
FashionCNN_Data = {'conv1':[], 'conv2':[], 'pool':[]}
FashionCNN_cdist = []
RSA_value = np.empty((3,6))

print("Done extracting relevant data")

f_layer = 0

for i in all_hooks:
    FashionCNN_Data[i] = torch.stack(stacked_activations[i]).detach().numpy()
    FashionCNN_cdist = cdist(FashionCNN_Data[i], FashionCNN_Data[i]) #include parameter here to toggle between Euclidean and Cosine
    row_indices, col_indices = np.triu_indices(1000, k = 1)
    Fashion_flat = FashionCNN_cdist[row_indices, col_indices]

    print(f"Running FashionCNN, {i}")

    c_layer = 0

    for j in all_hooksR:
        ResNet_Data[j] = torch.stack(stacked_activationsR[j]).detach().numpy()
        ResNet_cdist = cdist(ResNet_Data[j], ResNet_Data[j]) #include parameter here to toggle between Euclidean and Cosine
        Rrow_indices, Rcol_indices = np.triu_indices(1000, k = 1)
        ResNet_flat = ResNet_cdist[Rrow_indices, Rcol_indices]

        corr_value, p_eff = stats.spearmanr(Fashion_flat, ResNet_flat)
        RSA_value[f_layer, c_layer] = corr_value
        c_layer+=1
    
    f_layer+=1

print("Aggregating")

sns.heatmap(
    RSA_value, 
    annot=True, 
    xticklabels= ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'], 
    yticklabels= ['conv1', 'conv2', 'avgpool'], 
    cmap="viridis"
)

print("Plotting")

plt.title("Representational Similarity Analysis (RSA)")
plt.xlabel("ResNet50 Layers")
plt.ylabel("FashionCNN Layers")
plt.tight_layout()
plt.show()




