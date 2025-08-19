import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # Flatten label and prediction tensors
        # .view(-1) makes [batch_size, channels, height, width] turn into
        # a one dimensional vector of size batch_size*channels*height*width
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Ensurance of compatibility with the sigmoid output (values in [0,1])
        targets = (targets - targets.min())/(targets.max()-targets.min()) 
        
        # Calculation of Dice Loss
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, weight=None):

        # Flatten label and prediction tensors
        # .view(-1) makes [batch_size, channels, height, width] turn into
        # a one dimensional vector of size batch_size*channels*height*width
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        if weight != None:
            weight = weight.view(-1)

        targets = (targets - targets.min())/(targets.max()-targets.min())

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        if weight != None:
            # Weight must be a tensor of same shape as
            # input/target: 512x512
            
            BCE = F.binary_cross_entropy(inputs, targets, weight=weight, reduction='mean')
        else:    
            BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, standard_weight = 0):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.standard = standard_weight

    def forward(self, inputs, targets, weight=None):
        """
        inputs: predicted probabilities (after sigmoid), shape (N, 1, H, W) or (N,)
        targets: ground truth values in [0, 1], same shape as inputs
        weight: optional tensor of pixel-wise weights, same shape as inputs
        """

        # Flatten label and prediction tensors
        # .view(-1) makes [batch_size, channels, height, width] turn into
        # a one dimensional vector of size batch_size*channels*height*width
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        if weight != None:
            weight = weight.view(-1)

        # Compute BCE
        if weight is not None:
            loss = F.binary_cross_entropy(inputs, targets, weight=weight, reduction='mean')
        else:
            loss = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return loss