# import tensorflow as tf

from umap import UMAP

umap = UMAP(n_components=2,n_neighbors=50,random_state=2023).fit(coordinates)
print(umap)