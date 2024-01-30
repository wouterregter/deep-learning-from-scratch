import numpy as np

class DataLoaderScratch:
    def __init__(self, X, y, batch_size=64, shuffle=True):
        """
        Custom DataLoader for batching and iterating over a dataset.

        :param X: Input features, can be a list, numpy array, or PyTorch tensor.
        :param y: Labels corresponding to input features.
        :param batch_size: Number of samples per batch.
        :param shuffle: Whether to shuffle the data at the beginning of each iteration.
        :param transform: Optional transform to be applied on each batch.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        """Returns the number of batches in the DataLoader."""
        return int(np.ceil(len(self.X) / self.batch_size))

    def __iter__(self):
        """Iterator to generate data batches."""
        n_samples = len(self.X)
        indices = np.arange(n_samples)

        if self.shuffle:
            indices = np.random.permutation(indices)

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]

            yield X_batch, y_batch