import torch 
from torch import nn

class MSELossScratch(nn.Module):
    def __init__(self):
        super(MSELossScratch, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean((y_pred - y_true) ** 2)
        return loss


class CrossEntropyScratch(nn.Module):
    def __init__(self, num_classes):
        super(CrossEntropyScratch, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        y_one_hot = nn.functional.one_hot(y_true, self.num_classes)
        loss = -(y_one_hot * torch.log(y_pred)).sum(axis=1).mean()
        return loss