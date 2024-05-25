
import torch
import torchvision
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class SimCLR(nn.Module): 
    def __init__(self): 
        super(SimCLR, self).__init__()
        self.resnet = Net()
        self.head = ProjectionHead()

    def forward(self, x1, x2, t=0.1): 
        h1 = self.resnet(x1)
        h2 = self.resnet(x2)

        ## the WCL module applies F.normalize to z1, z2
        ## maybe for computational complexity? 
        z1 = self.head(h1)
        z2 = self.head(h2)     #dim = batch_size * embedding_size(128)
        
        # N = batch_size * 2
        # positive pairs identified by new labels (using index)
        z = torch.cat((z1, z2))     #dim = N * embedding_size(128)
        indices = torch.arange(0, z1.size(0)) 
        labels = torch.cat((indices, indices))      #dim = N
        return z, labels

class ProjectionHead(nn.Module): 
    def __init__(self, dim_in=2048, dim_out=128, dim_hidden=2048): 
        super(ProjectionHead, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden)
        self.linear3 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(self.bn1(x))
        x = self.linear2(x)
        x = F.relu(self.bn2(x))
        x = self.linear3(x)
        return x   

class Net(nn.Module): 
    '''Resnet class'''
    
    def __init__(self): 
        super(Net, self).__init__()
        
        # replace the first layer with a smaller conv layer [kernel size = 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,    # In: 3 channels
                               stride=1, padding=1,     # Out: 64 channels
                               bias = False)
        
        # remove the final fc (linear) layer so the output is of size 2048
        layers = list(models.resnet50(weights=None).children())[1:-1] # no weights
        self.middle = nn.Sequential(*layers) 

    def forward(self, x):  # [N * C * H * W]
        x = self.conv1(x)  #  [N * 64 * H * W]
        x = self.middle(x)  # [N * 2048] 2048 is resnet hidden dim
        return x.view(x.shape[0], -1) # [N * 2048]

