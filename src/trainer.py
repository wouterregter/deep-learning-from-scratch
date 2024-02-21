import torch
import matplotlib.pyplot as plt

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
                print(f"Validation {name}: {value:.4f}")

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