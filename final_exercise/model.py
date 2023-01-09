import torch
from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability 
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('Expected input to a 3D tensor') # pragma: no cover
        if x.shape[1] != 28 or x.shape[2] != 28:
            raise ValueError('Expected each sample to have shape [28, 28]') # pragma: no cover
        # x needs to be float to match with nn.Linear
        x = x.to(torch.float32)
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        # Output
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
