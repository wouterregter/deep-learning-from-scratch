import torch

class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        _, predicted = torch.max(outputs, 1)
        self.correct += (predicted == targets).sum().item()
        self.total += targets.size(0)

    def compute(self):
        return self.correct / self.total if self.total else 0

    def reset(self):
        self.correct = 0
        self.total = 0