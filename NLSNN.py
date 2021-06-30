import numpy as np

class NLSNN:
    def __init__(self, nneuron):
        self.NNeuron = nneuron
        self.Connections = np.random.uniform(-1, 1, (self.NNeuron, self.NNeuron))
        self.NeuronStates = np.zeros((self.NNeuron,))

        self.FireCount = np.zeros((self.NNeuron,))
    
    def Tick(self):
        buffer = np.copy(self.NeuronStates)
        fired = np.zeros_like(self.NeuronStates)
        for i in range(self.NNeuron):
            if self.NeuronStates[i] > 1.:
                self.FireCount[i] += 1
                fired[i] += 1.
                buffer[i] = 0.
                buffer = np.add(self.Connections[i, :], buffer)
        self.NeuronStates = buffer
        return buffer #fired
    
    def FireNeurons(self, indices):
        buffer = np.copy(self.NeuronStates)
        for i in range(len(indices)):
            self.FireCount[indices[i]] += 1
            buffer[indices[i]] = 0.
            buffer = np.add(self.Connections[indices[i], :][0], buffer)
        self.NeuronStates = buffer
        return buffer
    
    def ClearFeedback(self):
        self.FireCount = np.zeros((self.NNeuron,))

    def CreateFeedback(self, index, error, learning):
        #adjust for actual firing #TODO: test if not needed or performs better witout
        weightedConnections = self.Connections #np.tile(self.FireCount, (self.NNeuron, 1)) * self.Connections
        connectionErrors = np.zeros_like(self.Connections)
        for i in range(self.NNeuron):
            for j in range(self.NNeuron):
                connectionErrors[i, j] = weightedConnections[i, j] * error * learning
        return connectionErrors

    def RecursiveFeedback(self, index, error, learning, depth):
        connectionErrors = np.zeros_like(self.Connections)
        for i in range(self.NNeuron):
            connectionErrors[index, i] = self.Connections[index, i] * error# * learning
        if depth <= 0:
            return connectionErrors
        return np.add.reduce(np.array([self.RecursiveFeedback(i, connectionErrors[index, i], learning, depth - 1) for i in range(self.NNeuron)]))

    def ResetActivations(self):
        self.NeuronStates = np.zeros((self.NNeuron,))

    def WholePass(self):
        for i in range(self.NNeuron - 1):
            self.Tick()
        return self.Tick()
    
    def Learn(self, neuronErrors, learning):
        totalError = np.zeros_like(self.Connections)
        for i in range(len(neuronErrors)):
            error = neuronErrors[i]
            totalError -= self.RecursiveFeedback(i, error, learning, 3)
        self.ClearFeedback()
        self.Connections += totalError * learning
        return totalError
    
    def Train(self, FireList, Expected, learning, reset = False):
        totalError = 0
        for i in range(FireList.shape[0]):
            if reset:
                self.ResetActivations()
            self.FireNeurons(np.argwhere(FireList[i, :] == 1.))
            outputs = self.WholePass()
            errors = np.zeros_like(self.NeuronStates)
            for j in range(self.NNeuron):
                if Expected[i, j] == None:
                    errors[j] = 0
                else:
                    errors[j] = Expected[i, j] - outputs[j]
            totalError += sum(abs(errors))
            self.Learn(errors, learning)
        return totalError