import math
import numpy as np
import matplotlib.pyplot as plot

class neuron:
    def __init__(self):
        self.C = 100
        self.vr = -60
        self.vt = -40
        self.k = .7

        self.a = .03
        self.b = -2
        self.c = -50
        self.d = 100

        self.vpeak = 35

        self.T = 1000
        self.tau = 1
        self.n = round(self.T / self.tau)
        self.v = self.vr * np.ones((self.n, 1), float)
        self.u = 0 * self.v
        self.I = np.concatenate((np.zeros((int(0.1 * self.n), 1), float), 70 * np.ones((int(0.9 * self.n), 1), float)))

    def run(self):
        for i in range(0, self.n - 1):
            self.v[i + 1] = self.v[i] + self.tau * (self.k * (self.v[i] - self.vr) * (self.v[i] - self.vt) - self.u[i] + self.I[i]) / self.C
            self.u[i + 1] = self.u[i] + self.tau * self.a * (self.b * (self.v[i] - self.vr) - self.u[i])

            if self.v[i + 1] >= self.vpeak:
                self.v[i] = self.vpeak
                self.v[i + 1] = self.c
                self.u[i + 1] = self.u[i + 1] + self.d

        plot.plot(self.tau * list(range(1, self.n + 1)), self.v)
        plot.show()

if __name__ == "__main__":
    myneuron = neuron()
    myneuron.vr = -30
    myneuron.run()