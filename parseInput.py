import struct
# import neural
import numpy as np
import pylab as pl

class Loader:
    def __init__(self, isTraining):
        if isTraining:
            labels = open("MNIST-training/train-labels.idx1-ubyte", "rb");
            images = open("MNIST-training/train-images.idx3-ubyte", "rb");
        else:
            labels = open("MNIST-test/t10k-labels.idx1-ubyte", "rb");
            images = open("MNIST-test/t10k-images.idx3-ubyte", "rb");
        self.labels = labels;
        self.images = images;
        magic, nimages = struct.unpack(">LL", labels.read(8))
        magic2, nimages2, nrow, ncol = struct.unpack(">LLLL", images.read(16));
        print magic, nimages
        print magic2, nimages2, nrow, ncol
        self.nimages = nimages;
    def getDigits(self):
        for i in range(self.nimages):
            (lab,) = struct.unpack("B", self.labels.read(1));
            img = np.reshape(struct.unpack("784B", self.images.read(784)), (28, 28));
            yield lab, img;

if __name__ == "__main__":
    x = Loader();
    for lab, img in x.getDigits():
        pl.imshow(img);
        print lab;    
        pl.show();
        raw_input("Press enter to continue")


