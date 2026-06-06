# FashionCNN's centroids are compared with the embeddings
# found within the sentence transformer all-minilm-l6-v2
# Values obtained are used to plot a regplot (scatter plot + trendline)

from sentence_transformers import SentenceTransformer
from FashionCNN_Centcalc import class_names, final_comp
from scipy.spatial.distance import cdist 
from scipy import stats
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = model.encode(class_names)


fashdist = cdist(final_comp, final_comp)
row_indices, col_indices = np.triu_indices(10, k = 1)
fashion_flat = fashdist[row_indices, col_indices]

sidist  = cdist(embeddings, embeddings)
Srow_indices, Scol_indices = np.triu_indices(10, k = 1)
si_flat = sidist[Srow_indices, Scol_indices]


corr_value, p_eff = stats.spearmanr(fashion_flat, si_flat)

sns.regplot(
    x = fashion_flat,
    y = si_flat
)

print("Plotting")

plt.title(f"Representational Similarity Analysis (RSA), value: {corr_value}")
plt.xlabel("FashionCNN")
plt.ylabel("Sentence transformer")
plt.tight_layout()
plt.show()

# print(embeddings.shape)

# similarities = model.similarity(embeddings, embeddings)
# print(similarities)
