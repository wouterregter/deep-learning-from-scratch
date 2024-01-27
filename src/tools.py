import numpy as np
import torch
import matplotlib.pyplot as plt

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

class SGDScratch:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        with torch.no_grad():
            # Update parameters (gradient descent)
            for param in self.parameters:
                param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            # Zero gradients (if they exist)
            if param.grad is not None:
                param.grad.zero_()


class TrainerScratch:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, custom_metrics=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.custom_metrics = custom_metrics if custom_metrics else {}

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_dataloader:
            inputs, targets = batch

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, targets = batch

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                for name, metric in self.custom_metrics.items():
                    metric.update(outputs, targets)

        avg_loss = total_loss / len(self.val_dataloader)
        metrics_results = {name: metric.compute() for name, metric in self.custom_metrics.items()}
        return avg_loss, metrics_results

    def fit(self, num_epochs):
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            for name, value in val_metrics.items():
                print(f"{name}: {value:.4f}")

            # Reset custom metrics for next epoch
            for metric in self.custom_metrics.values():
                metric.reset()
        
        # Plot the training and validation losses
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

