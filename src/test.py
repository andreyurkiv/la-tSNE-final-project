from sklearn import datasets
from src.tSNE import tSNE

file = datasets.load_digits()
data = file.data
points = tSNE(out_dims=2, perplexity=20, n_iter=100).fit_transform(data)
print(points.shape)
