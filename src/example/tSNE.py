import numpy as np


class tSNE():
    def __init__(self, X=None, out_dims=2, perplexity=30.0, early_exaggeration=4.,
                 learning_rate=200.0, n_iter=1000, min_grad_norm=1e-07, random_seed=None):
        """t-distributed Stochastic Neighbor Embedding."""
        self.X = X
        self.out_dims = out_dims
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.min_grad_norm = min_grad_norm
        self.random_seed = random_seed

    def _calc_H_and_perplexity(self, D, beta):
        """ToDo"""
        P = np.exp(-D.copy() * beta)
        sum_P = sum(P)
        H = np.log(sum_P) + beta * np.sum(D * P) / sum_P
        P = P / sum_P
        return H, P

    def _pairwise_dist(self, X, tol=1e-5):
        """ToDo"""
        samples_size, in_dims = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((samples_size, samples_size))
        beta = np.ones((samples_size, 1))
        log_U = np.log(self.perplexity)

        for i in range(samples_size):
            if i % 10 == 0:
                print(f"Computing P-values for {i + 1}/{samples_size} point.\r", end='')

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax = np.inf
            D_i = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:samples_size]))]
            H, this_P = self._calc_H_and_perplexity(D_i, beta[i])

            # Evaluate whether the perplexity is within tolerance
            H_diff = H - log_U
            tries = 0
            while np.abs(H_diff) > tol and tries < 50:

                # If not, increase or decrease precision
                if H_diff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                H, this_P = self._calc_H_and_perplexity(D_i, beta[i])
                H_diff = H - log_U
                tries += 1

            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:samples_size]))] = this_P

        # Return final P-matrix
        print()
        print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P


    def _tsne(self, X):
        if self.random_seed:
            np.random.seed(self.random_seed)
        X = X / np.max(np.abs(X))  # to avoid div by zero later
        # recomended to use PCA for data preprocessing to reduce dimensionality
        # if in_dims is quite a big number (e. g. 100+)
        # X = self.pca(X, initial_dims).real
        samples_size, in_dims = X.shape
        initial_momentum = 0.5
        final_momentum = 0.8
        eta = 500
        min_gain = 0.01
        Y = np.random.randn(samples_size, self.out_dims)
        dY = np.zeros((samples_size, self.out_dims))
        iY = np.zeros((samples_size, self.out_dims))
        gains = np.ones((samples_size, self.out_dims))

        # Compute P-values
        P = self._pairwise_dist(X, 1e-5)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        P = P * self.early_exaggeration
        P = np.maximum(P, 1e-12)

        # Run iterations
        for iter in range(self.n_iter):

            # Compute pairwise affinities
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(samples_size), range(samples_size)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # Compute gradient
            PQ = P - Q
            for i in range(samples_size):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (self.out_dims, 1)).T * (Y[i, :] - Y), 0)

            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                    (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (samples_size, 1))

            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q))
                print(f"Iteration {iter + 1}: error (cost) is {round(C, 5)}\r", end='')

            # Stop early exaggeration for P-values
            if iter == 100:
                P = P / self.early_exaggeration

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
