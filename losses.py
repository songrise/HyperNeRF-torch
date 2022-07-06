import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['coarse']['rgb'], targets)
        if 'fine' in inputs:
            loss = loss + self.loss(inputs['fine']['rgb'], targets)

        return loss
               

loss_dict = {'mse': MSELoss}