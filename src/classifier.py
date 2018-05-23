from akhsabrg_tests.cnn_combined import COMBINED_CNN
import numpy as np

class Classifier:

    def __init__(self):
        self.network = COMBINED_CNN()
        #self.network.train(1)

    def forward(self, img):
        img = img.astype(np.float32)
        guessed = self.network.forward(img)
        final_class = self.network.class2char(guessed)
        return final_class
