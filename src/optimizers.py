import torch

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