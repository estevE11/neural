from torch import nn

class BG(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(2, 2)
        self.act2 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.act2(x)
        return x