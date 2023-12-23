from tqdm import trange
import numpy as np


from dataset import fetch_mnist
X_train, Y_train, X_test, Y_test = fetch_mnist()

from torch import nn

class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 128)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        return x
    

import torch
import math
import random
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    model = MnistNet()
    batch_size = 32 # actuially makes it better

    # train
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())
    for i in (t:=trange(400)):
        samp = np.random.randint(0, X_test.shape[0], size=(batch_size))
        inpt = torch.tensor(X_test[samp].reshape(-1, 28*28)).float()
        target = torch.tensor(Y_test[samp])
        optim.zero_grad()
        guess = model(inpt)
        loss = loss_function(guess, target)
        loss.backward()
        optim.step()
        t.set_description(f"Loss {loss.item():.2f}")


    corr = 0
    samples = 1000
    for i in (t := trange(samples)):
        X = torch.tensor(X_test[i].reshape((-1, 28*28))).float()
        Y = Y_test[i]
        guess = torch.argmax(model(X))
        if Y == guess: corr += 1
        t.set_description(f"Accuracy {(corr/max(i, 1)*100):.2f}%")