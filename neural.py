#!/usr/bin/env python

"""
Python source code - replace this with a description of the code and write the code below this text.
"""
import numpy as np
from scipy.special import expit
import parseInput
import matplotlib.pyplot as plt
import json

def sigmoid(z):
    if(z > 400):
        return 1;
    if(z < -400):
        return 0;
    return expit(z);
sigmoid_vec = np.vectorize(sigmoid)

class Layer:
    def __init__(self, size=1, isSource=False, prev = None):
        self.size = size;
        self.values = np.zeros((size, 1));
        if isSource:
            pass
        else:
            self.prev = prev;
            self.params = np.random.randn(size, prev.size);
            self.const = np.random.randn(size, 1);
            self.nextParams = np.zeros((size, prev.size));
            self.nextConst = np.zeros((size, 1));
            self.testCount = 0;
        self.isSource = isSource;
    
    def setValues(self, values):
        self.values = np.reshape(values, (self.size, 1));
        
    def eval(self):
        if(self.isSource):
            return;
        self.prev.eval();
        self.values = sigmoid_vec(np.add(np.dot(self.params, self.prev.values), self.const));
        
    def comDeriv(self, derivs):
        if(self.isSource):
            return;
        self.derivs = derivs; #derivs is a column vector
        self.prev.comDeriv(
            np.transpose(
                np.dot(
                    np.transpose(
                        np.multiply(
                            np.multiply(self.values, np.subtract(1, self.values)),
                            derivs
                        )
                    ),
                    self.params
                )
            )
        )

    def updateValues(self, evoRate):
        if(self.isSource):
            return;
        self.nextParams = np.subtract(self.nextParams, 
            np.multiply(
                np.dot(
                    np.multiply(
                        np.multiply(self.values, np.subtract(1, self.values)),
                        self.derivs
                    ),
                    np.transpose(self.prev.values)
                ),
                evoRate
            )
        );

        self.nextConst = np.subtract(self.nextConst, 
            np.multiply(
                np.multiply(
                    np.multiply(self.values, np.subtract(1, self.values)),
                    self.derivs
                ),
                evoRate
            )
        );
        
        self.testCount += 1;

        self.prev.updateValues(evoRate);
    def nextStep(self):
        if(self.isSource):
            return;
        self.const = np.add(self.const, np.true_divide(self.nextConst, self.testCount));
        self.params = np.add(self.params, np.true_divide(self.nextParams, self.testCount));
        self.nextParams = np.zeros((self.size, self.prev.size));
        self.nextConst = np.zeros((self.size, 1));
        self.testCount = 0;
        self.prev.nextStep();
class Digit:
    def __init__(self, gg):
        self.label, self.pixels= gg;
        self.pixels = np.true_divide(self.pixels, 256);
    def __str__(self):
        return str(self.label);
if __name__ == "__main__":
    print "hello!";
    loader = parseInput.Loader(isTraining = True);
    # batchSize = 300;
    # w = [Digit(loader.getDigits().next()) for x in range(batchSize)];
    w = [Digit(x) for x in loader.getDigits()];
    inp = Layer(size = 28*28, isSource = True);
    med = Layer(size = 80, isSource = False, prev = inp);
    out = Layer(size = 10, isSource = False, prev = med);
    score = [];
    pr = np.array([x for x in range(loader.nimages)]);
    for repeat in range(300):
        score.append(0);
        np.random.shuffle(pr);
        for gg in range(0, 500):
            i = w[pr[gg]];
            inp.setValues(i.pixels)
            out.eval();
            if(out.values.tolist().index(max(out.values)) == i.label):
                score[-1] += 1;
            derivs = np.subtract(out.values, [[int(j == i.label)] for j in range(10)])
            out.comDeriv(derivs);
            out.updateValues(0.3);
            out.nextStep();
        if repeat%100 == 99:
            plt.plot(score);
            plt.show();

    plt.plot(score);
    plt.show();
    print score[-1]
    with open('bigparams-80.json', 'w') as outfile:
        outfile.write(json.dumps({
            "out.params": out.params.tolist(),
            "out.const":  out.const.tolist(),
            "med.params": med.params.tolist(),
            "med.const":  med.const.tolist(),
        }))
        print "wrote stuff"






