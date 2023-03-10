import torch
import numpy as np
from model import BG

if __name__ == "__main__":
    model = BG()
    model.load_state_dict(torch.load("models/color_2.pt"))
    model.eval()

    color = torch.tensor(np.array([27/255, 242/255, 84/255])).float()
    guess = model(color)
    print(guess)

    