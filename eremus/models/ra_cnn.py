import os
import sys
import math
import time
import random
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import torch; torch.utils.backcompat.broadcast_warning.enabled = True

# ASYMMETRIC DIFFERENTIAL LAYER
def Adl_2d(x):
    """
    Asymmetric Differential Layer (ADL) in 2D.
    
    Given a matrix IN of shape [H, W], returns a matrix OUT of shape [H , integer_part(W/2)].
    OUT(i, j) = IN(i, j) - IN(i, W + 1 - j).
    In short, The input matrix is folded in half along width dimension, 
    and the overlapped elements are subtracted beetween them.
    
    Params:
        - x: torch.Tensor, input matrix (size HxW)
    Returns
        torch.Tensor, ADL output matrix (size Hx(W/2))
    """
    # Given a matri
    w = x.size(-1)
    w_2 = int(w/2)
    sx = x[:, :w_2]
    dx = x[:, -w_2:]
    return sx - dx.flip(-1)

def Adl_3d(x):
    """
    Asymmetric Differential Layer (ADL) in 3D.
    
    Given a matrix IN of shape [F, H, W], returns a matrix OUT of shape [F, H , integer_part(W/2)].
    OUT(k, i, j) = IN(k, i, j) - IN(k, i, W + 1 - j).
    In short, The input matrix is folded in half along width dimension, 
    and the overlapped elements are subtracted beetween them.
    ADL in 3D is equivalent to perform ADL in 2D F times, one for HxW matrix in input.
    
    Params:
        - x: torch.Tensor, input matrix (size FxHxW)
    Returns
        torch.Tensor, ADL output matrix (size FxHx(W/2))
    """
    w = x.size(-1)
    w_2 = int(w/2)
    sx = x[:, :, :w_2]
    dx = x[:, :, -w_2:]
    return sx - dx.flip(-1)

def Adl_4d(x):
    """
    Asymmetric Differential Layer (ADL) in 4D.
    
    Given a matrix IN of shape [B, F, H, W], returns a matrix OUT of shape [B, F, H , integer_part(W/2)].
    OUT(b, k, i, j) = IN(b, k, i, j) - IN(b, k, i, W + 1 - j).
    In short, The input matrix is folded in half along width dimension, 
    and the overlapped elements are subtracted beetween them.
    ADL in 4D is equivalent to perform ADL in 3D B times, one for FxHxW matrix in input.
    
    Params:
        - x: torch.Tensor, input matrix (size BxFxHxW)
    Returns
        torch.Tensor, ADL output matrix (size BxFxHx(W/2))
    """
    w = x.size(-1)
    w_2 = int(w/2)
    sx = x[:, :, :, :w_2]
    dx = x[:, :, :, -w_2:]
    return sx - dx.flip(-1)

class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        args_defaults=dict(num_channels=32, num_classes=4, verbose=False)
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
    
        # TEMPORAL FEATURES EXTRACTOR
        self.temporal = nn.Sequential(
            
            # Layer 1
            nn.Conv3d(1, 32, (1, 1, 5), stride=(1, 1, 2)), #padding = (0, 0, 2),
            nn.ELU(),
            nn.BatchNorm3d(32, False),
            nn.Dropout(0.25),
            
            # Layer 2
            #nn.ZeroPad2d((16, 17, 0, 1))
            nn.Conv3d(32, 32, (1, 1, 3), stride=(1, 1, 2)), #padding = (0, 0, 1),
            nn.ELU(),
            nn.BatchNorm3d(32, False),
            nn.Dropout(0.25),
            #nn.MaxPool2d(2, 4)

            # Layer 3
            #nn.ZeroPad2d((2, 1, 4, 3))
            nn.Conv3d(32, 32, (1, 1, 3), stride=(1, 1, 2)), #padding = (0, 0, 1),
            nn.ELU(),
            nn.BatchNorm3d(32, False),
            nn.Dropout(0.25),
            #nn.MaxPool2d((2, 4))

            # Layer 4
            nn.Conv3d(32, 32, (1, 1, 16), stride=(1, 1, 4)),
            nn.ELU(),
            nn.BatchNorm3d(32, False),
            nn.Dropout(0.25),

            # Layer 5
            nn.Conv3d(32, 32, (1, 1, 36)),
            nn.ELU(),
            nn.BatchNorm3d(32, False),
            nn.Dropout(0.25)
        )
        
        # REGIONAL FEATURES EXTRACTOR
        self.regional = nn.Sequential(
            
            # Layer 1
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32, False),
            nn.Dropout(0.25),

            # Layer 2
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32, False),
            nn.Dropout(0.25)
        )
        
        # ASYMMETRIC FEATURES EXTRACTOR
        self.asymmetric = nn.Sequential(
            
            nn.Conv2d(32, 64, 1, padding=0),
            nn.ELU(),
            nn.BatchNorm2d(64, False),
            nn.Dropout(0.25)
        )
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 1280 timepoints. 
        self.fully_connected = nn.Sequential(
            
            nn.Linear(4896, 128),
            nn.Sigmoid(),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=1)
        )
        

    def forward(self, x):
        # Add feature dimension [B, H, W, D] --> [B, F, H, W, D]
        x = x.unsqueeze(1)

        # Temporal Features Extractor
        if self.verbose:
            print('\n[TEMPORAL EXTRACTOR] Input size:', end='\t')
            print(x.size())
        x = self.temporal(x)
        if self.verbose:
            print('[TEMPORAL EXTRACTOR] Output size:', end='\t')
            print(x.size())
        
        # Depth is collapsed to 1: delete D dimension [B, F, H, W, 1] --> [B, F, H, W]
        x.squeeze_(-1)
        
        # Regional Features Extractor
        if self.verbose:
            print('\n[REGIONAL EXTRACTOR] Input size:', end='\t')
            print(x.size())
        x0 = self.regional(x)
        if self.verbose:
            print('[REGIONAL EXTRACTOR] Outpute size:', end='\t')
            print(x0.size())
        
        # Asymmetric Features Extractor
        if self.verbose:
            print('\n[ASYMMETRIC EXTRACTOR] Input size:', end='\t')
            print(x.size())
        x1 = self.asymmetric(Adl_4d(x))
        if self.verbose:
            print('[ASYMMETRIC EXTRACTOR] Output size:', end='\t')
            print(x1.size())

        
        # FC Layer
        # Collpase regional features, keep batch dimension [B, F, H, W] --> [B, F]
        x0 = x0.view(x0.shape[0], -1)
        # Collpase asymmetric features, keep batch dimension [B, F, H, W] --> [B, F]
        x1 = x1.view(x1.shape[0], -1)
        # Contatenate regional and asymmetric features along features dimension
        x = torch.cat((x0, x1), 1)
        if self.verbose:
            print('\n[FULLY CONNECTED LAYER] Input size:', end='\t')
            print(str(x0.size(1)) + '|' + str(x1.size(1)), end=' --> ')
            print(x.size())
        x = self.fully_connected(x)
        if self.verbose:
            print('[FULLY CONNECTED LAYER] Output size:', end='\t')
            print(x.size())
        return x

if __name__ == "__main__":
    net = Model(dict(num_channels=32, num_classes=22, verbose=True)).cuda(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(dev)  