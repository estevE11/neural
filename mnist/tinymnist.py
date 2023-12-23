from tqdm import trange
from tinygrad import Tensor, nn
import numpy as np
from dataset import fetch_mnist

class TinyMNIST:
    def __init__(self):
        self.l1 = nn.Linear(784, 128, bias=True)
        self.l2 = nn.Linear(128, 10, bias=True)
    
    def __call__(self, x):
        x = x.flatten(1)
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        return x


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = fetch_mnist()

    # train
    model = TinyMNIST()
    optim = nn.optim.Adam([model.l1.weight, model.l2.weight], lr=0.001)
    
    BS = 32
    with Tensor.train():
        for i in (t := trange(400)):
            samp = np.random.randint(0, X_train.shape[0], size=(BS, 1))
            ipt = Tensor(X_train[samp]).float()
            samp = samp.reshape(BS)
            target = Tensor(Y_train[samp]).float()
            optim.zero_grad()
            out = model(ipt)
            loss = out.sparse_categorical_crossentropy(target)
            loss.backward()
            optim.step()
            t.set_description(f"Loss {loss.item():.2f}")

    # test accuracy
    corr = 0
    samples = 1000
    for i in (t := trange(samples)):
        samp = np.random.randint(0, X_test.shape[0], size=(1, 1))
        X = Tensor(X_test[samp]).float()
        Y = Y_test[samp][0][0]
        guess = model(X).argmax()
        if Y == guess.item(): corr += 1
        t.set_description(f"Accuracy {(corr/max(i, 1)*100):.2f}%")


