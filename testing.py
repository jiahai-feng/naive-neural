import neural
import parseInput
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    inp = neural.Layer(size = 28*28, isSource = True);
    med = neural.Layer(size = 80, isSource = False, prev = inp);
    out = neural.Layer(size = 10, isSource = False, prev = med);
    score = 0;
    loader = parseInput.Loader(isTraining = False);
    params = json.loads(open('bigparams-80.json', "r").read());
    med.params = np.array(params['med.params']);
    out.params = np.array(params['out.params']);
    med.const  = np.array(params['med.const']);
    out.const  = np.array(params['out.const']);
    for x in loader.getDigits():
        y = neural.Digit(x);
        inp.setValues(y.pixels);
        out.eval();
        if(out.values.tolist().index(max(out.values)) == y.label):
            score += 1;
    # plt.plot(score);
    # plt.show();
    print "Score: ", score;
