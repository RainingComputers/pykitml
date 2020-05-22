import numpy as np

from ._classifier import Classifier
from ._functions import tanh, sigmoid, tanh_prime, sigmoid_prime     

__all__ = ['LSTM']

# Enums
Wz, Wi, Wf, Wo, Rz, Ri, Rf, Ro, Bz, Bi, Bf, Bo = list(range(12)) 

class LSTM(Classifier):
    def __init__(self, layer_sizes):
        self.t = 0
        self._lsizes = layer_sizes
        self.nlayers = len(layer_sizes)
        self._bsize = None

        # Initialize LSTM gates with list for each layer
        [self.z_ws, self.z, self.i_ws, self.i, self.f_ws, self.f, self.c, 
            self.o_ws, self.o, self.y] = [[np.array([])]*self.nlayers]*10

        # Initialize LSTM weights
        [w_z, w_i, w_f, w_o, r_z, r_i, r_f, r_o, b_z, 
            b_i, b_f, b_o] = [[np.array([])]*self.nlayers for i in range(12)]
        # Loop through each layer and initialize random weights and b_z
        for l in range(1, self.nlayers):
            lsize = layer_sizes[l]
            input_lsize = layer_sizes[l-1]
            eps = np.sqrt(6 / (lsize + input_lsize))
            w_z[l], w_i[l], w_f[l], w_o[l] = [np.random.uniform(-eps, eps, (lsize, input_lsize)) for i in range(4)]
            r_z[l], r_i[l], r_f[l], r_o[l] = [np.random.uniform(-eps, eps, (lsize, lsize)) for i in range(4)]
            b_z[l], b_i[l], b_f[l], b_o[l] = [np.random.uniform(-eps, eps, (lsize)) for i in range(4)]
        # Put parameters in numpy dtype=object array
        self._params = np.array(
            [np.array(w_z, dtype=object), np.array(w_i, dtype=object),
                np.array(w_f, dtype=object), np.array(w_o, dtype=object),
                np.array(r_z, dtype=object), np.array(r_i, dtype=object),
                np.array(r_f, dtype=object), np.array(r_o, dtype=object),
                np.array(b_z, dtype=object), np.array(b_i, dtype=object),
                np.array(b_f, dtype=object), np.array(b_o, dtype=object)],
            dtype=object
        )  

    @property
    def _out_size(self):
        return self._lsize[-1]

    def _init_gates(self, batch_size):
        self.y[0] = np.zeros((batch_size, self._lsizes[0]))

        self._bsize = batch_size
        for l in range(1, self.nlayers): 
            [self.z_ws[l], self.z[l], self.i_ws[l], self.i[l], self.f_ws[l], 
                self.f[l], self.c[l], self.o_ws[l], self.o[l], 
                self.y[l]] = [np.zeros((batch_size, self._lsizes[l])) for i in range(10)]


    def _reset_time():
        self.t = 0

    def _reset_state():
        pass

    def _reset():
        pass

    def feed(self, input_data):
        t = self.t%self._bsize
        t_1 = (self.t-1)%self._bsize

        self.y[0][t] = input_data
        for l in range(1, self.nlayers-1):
            # Block input
            self.z_ws[l][t] = self.y[l-1][t]@self._params[Wz][l].T + self.y[l][t_1]@self._params[Rz][l].T + self._params[Bz][l]
            self.z[l][t] = tanh(self.z_ws[l][t])
            # Input gate
            self.i_ws[l][t] = self.y[l-1][t]@self._params[Wi][l].T + self.y[l][t_1]@self._params[Ri][l].T + self._params[Bi][l]
            self.i[l][t] = sigmoid(self.i_ws[l][t])
            # Forget gate
            self.f_ws[l][t] = self.y[l-1][t]@self._params[Wf][l].T + self.y[l][t_1]@self._params[Rf][l].T + self._params[Bf][l]
            self.f[l][t] = sigmoid(self.f_ws[l][t])
            # Output gate
            self.o_ws[l][t] = self.y[l-1][t]@self._params[Wo][l].T + self.y[l][t_1]@self._params[Ro][l].T + self._params[Bo][l]
            self.o[l][t] = sigmoid(self.o_ws[l][t])
            # Cell state
            self.c[l][t] = self.z[l][t]*self.i[l][t] + self.c[l][t_1]*self.f[l][t]
            # Block output
            self.y[l][t] = tanh(self.c[l][t])*self.o[l][t]

        self.t += 1


    def get_output(self):
        t_1 = (self.t-1)%self._bsize
        return self.y[-1][t_1]

    def _backpropagate(self):
        # Initialize LSTM gates' gradients
        [dz_ws, dz, di_ws, di, df_ws, df, 
            dc, do_ws, do, dy] = [[np.array([])]*dnlayers]*10

        # Initialize LSTM weights' gradients
        [dw_z, dw_i, dw_f, dw_o, dr_z, dr_i, dr_f, dr_o, db_z, 
            db_i, db_f, db_o] = [[np.array([])]*self.nlayers for i in range(12)]

        #for t in range(self.t):
            

    def train(self, training_data, targets, batch_size, epochs, optimizer,
            testing_data=None, testing_targets=None, testing_freq=1, decay_freq=1):
        pass

    def plot_performance():
        pass

    def cost():
        pass

