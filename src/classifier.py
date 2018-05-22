from akhsabrg_tests.nn_mnist import NN_MNIST
from akhsabrg_tests.cnn_combined import COMBINED_CNN
import numpy as np

class Classifier:

    def __init__(self):
        self.network = COMBINED_CNN()
        #self.network.train(1)

    def forward(self, img):
        img = img.astype(np.float32)
        fixed_orientation = np.transpose(img)
        fixed_shape = np.reshape(fixed_orientation, (1, 28*28))
        guessed = self.network.forward(fixed_shape)
        final_class = self.network.class2char(guessed)
        return final_class
