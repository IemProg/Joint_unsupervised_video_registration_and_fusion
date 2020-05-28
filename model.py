import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

class AffineNet(nn.Module):
    def __init__(self):
        super(AffineNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 80, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(80, 100, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(True),
            nn.Linear(100, 50), # add for model3
            nn.ReLU(True),     # add for model3
            nn.Linear(50, 3 * 2)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    # Spatial transformer network forward function
    def stn(self, x):
        #print("Starting stn, x: ", x.size())
        xs = self.localization(x)
        #print("XS", xs.size())
        xs = xs.view(-1, 200)  
        #print("XS", xs.size())
        theta = self.fc_loc(xs)
        #print("theta: ", theta.size())
        theta = theta.view(-1, 2, 3)
        #print("theta: ", theta.size())
        #print("X: ", x.size())

        grid = nnf.affine_grid(theta, x.size())
        #print("Grid: ", grid.size())
        x = nnf.grid_sample(x, grid)

        return x, grid
    
    def forward(self, x):#src, tgt):
        #x = torch.cat([src, tgt], dim=1)
        # transform the input
        x = self.stn(x)

        return x #F.log_softmax(x, dim=1)
