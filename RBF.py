
from scipy import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt

class RBF:

    def init(self):

        self.centers = random.uniform(-2, 2, 10)
        self.W = random.random((10))

    def radialFunc(self, x, t):
        return exp(-1.0 * (norm(x - t) ** 2))

      # calculate activations of RBFs
    def calcAct(self, X):
        G = zeros((X.shape[0], 10), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self.radialFunc(c, x)
        return G

    def train(self, X, Y):
        rnd_idx = random.permutation(X.shape[0])[:10]
        self.centers = [X[i, :] for i in rnd_idx]
        G = self.calcAct(X)
        self.W = dot(pinv(G), Y)

    def predict(self, X):

        G = self.calcAct(X)
        Y = dot(G, self.W)
        return Y



if __name__ == '__main__':

    n = 100

    x = mgrid[-2:2:complex(0, n)].reshape(n, 1)
    y = x*x
    rbf = RBF()
    rbf.train(x, y)
    z = rbf.predict(x)
    plt.figure()
    plt.plot(x, y, 'go' )
    plt.plot(x, z, 'yo')
    plt.show()
    print("green shows original X^2 and yellow shows predicted X^2")

