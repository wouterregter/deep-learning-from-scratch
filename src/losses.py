import torch 
from torch import nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        num_classes = y_pred.size(1)
        y_one_hot = nn.functional.one_hot(y_true, num_classes)
        loss = -(y_one_hot * torch.log(y_pred)).sum(axis=1).mean()
        return loss