from NLSNN import NLSNN
import numpy as np
import matplotlib.pyplot as plt

for j in range(1):
    print(j)
    NNeuron = 10
    test = NLSNN(NNeuron)
    nTrain = 4
    inputs = np.zeros((nTrain, NNeuron))
    inputs[0, 0] = 1
    inputs[0, 1] = 1

    inputs[1, 0] = 0
    inputs[1, 1] = 0

    inputs[2, 0] = 1
    inputs[2, 1] = 0

    inputs[3, 0] = 0
    inputs[3, 1] = 1

    outputs = np.full((nTrain, NNeuron), None)
    outputs[0, 2] = 0
    outputs[0, 3] = 1

    outputs[1, 2] = 1
    outputs[1, 3] = 0

    outputs[2, 2] = 1
    outputs[2, 3] = 0

    outputs[3, 2] = 1
    outputs[3, 3] = 0

    data = []
    for i in range(25):
        #error = test.Train(inputs, outputs, .01)
        error = test.Train(inputs, outputs, 0.1, True)
        #print(".")
        data.append(error)
        #print(error)
    #if data[-1] > 1000:
        #continue
    print(data[-1])
    plt.plot(data)

plt.show()