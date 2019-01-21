import numpy as np

class tSNE():
    def __init__(self, X=None, out_dims=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, min_grad_norm=1e-07, random_seed=None):
        """t-distributed Stochastic Neighbor Embedding."""
        self.X = X
        self.out_dims = out_dims
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.min_grad_norm = min_grad_norm
        self.random_seed = random_seed

    def _tsne(self, X):
        samples_size, in_dims = X.shape
        if self.random_seed:
            np.random.seed(self.random_seed)
        # recomended to use PCA for data preprocessing to reduce dimensionality
        # if in_dims is quite a big number (e. g. 100+)
        Y = np.random.randn(samples_size, self.out_dims)
        # ToDo
        self.embeddings_ = Y
        return Y

    def fit(self, X, **kwargs):
        """Fit X into an embedded space"""
        self.X = X
        self._tsne(self.X)

    def fit_transform(self, X, **kwargs):
        """Fit X into an embedded space and return that transformed output in out_dim space."""
        self.fit(X, **kwargs)
        return self.embeddings_

    def __repr__(self):
        return (f't-SNE(self.out_dims={self.out_dims}, self.perplexity={self.perplexity},\n'
                f'      self.learning_rate={self.learning_rate}, self.n_iter={self.n_iter},\n'
                f'      self.min_grad_norm={self.min_grad_norm}, self.random_seed={self.random_seed})')
