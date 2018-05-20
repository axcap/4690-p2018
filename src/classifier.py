from akhsabrg_tests.nn_mnist import NN_MNIST
import numpy as np

class Classifier:

    def __init__(self):
        self.nn = NN_MNIST(model_path="res/model/nn_fnist/fnist_demo")
        self.nn.train(None)

    def forward(self, img):
        classes = self.nn.forward(np.reshape(img, (1, 28*28)))
        return np.argmax(classes)
