from copy import deepcopy

import numpy as np

from ._minimize_model import MinimizeModel
from ._classifier import Classifier
from ._regressor import Regressor

from . import _functions
from ._functions import tanh, sigmoid, tanh_prime, sigmoid_prime     
from ._functions import mse, mse_prime

__all__ = ['LSTM']

# Enums
Wz, Wi, Wf, Wo, Rz, Ri, Rf, Ro, Bz, Bi, Bf, Bo, OUT_W, OUT_B = list(range(14)) 

class LSTM(MinimizeModel, Classifier, Regressor):
    '''
    This class implements LSTM Network.
    '''
    def __init__(self, layer_sizes, output_activ_func='softmax', cost_func='cross_entropy'):
        '''
        Parameters
        ----------
        layer_sizes : list
            A list of integers describing the number of layers and the number of neurons in each
            layer. For e.g. :code:`[784, 100, 100, 10]` describes a network with one input
            layer having 784 neurons, two hidden LSTM layers having 100 neurons each and a 
            dense output layer with 10 neurons.
        output_activ_function : str
            Activation function to use for dense output layer.
            List of available activation functions:
            :code:`leakyrelu`, :code:`relu`, :code:`softmax`, :code:`tanh`, :code:`sigmoid`, :code:`identity`.
        cost_function : str
            List of available cost functions:
            :code:`mse` (Mean Squared Error), :code:`cross_entropy` (Cross Entropy), :code:`huber` (Huber loss).
        '''
        self.t = 1
        self._lsizes = layer_sizes
        self._nlayers = len(layer_sizes)
        self._bsize = None
        self._outputs = None
        self._batch_count = 0

        # Initialize output activation function and cost function
        self._output_activ_func = getattr(_functions, output_activ_func)
        self._output_activ_func_prime = getattr(_functions, output_activ_func+'_prime')
        self._cost_func = getattr(_functions, cost_func)
        self._cost_func_prime = getattr(_functions, cost_func+'_prime')

        # Initialize LSTM gates with list for each layer
        [self.z_ws, self.z, self.i_ws, self.i, self.f_ws, self.f, self.c, 
            self.o_ws, self.o, self.y, self.last_y, self.last_c] = \
                [[np.array([])]*(self._nlayers-1) for i in range(12)]
        
        # Initialize dense output layer activations
        self.out_ws = None
        self.out_a = None 

        # Initialize LSTM weights
        [w_z, w_i, w_f, w_o, r_z, r_i, r_f, r_o, b_z, 
            b_i, b_f, b_o] = [[np.array([])]*(self._nlayers-1) for i in range(12)]
        # Loop through each layer and initialize random weights and b_z
        for l in range(1, self._nlayers-1):
            lsize = layer_sizes[l]
            input_lsize = layer_sizes[l-1]
            eps = np.sqrt(6 / (lsize + input_lsize))
            w_z[l], w_i[l], w_f[l], w_o[l] = [np.random.uniform(-eps, eps, (lsize, input_lsize)) for i in range(4)]
            r_z[l], r_i[l], r_f[l], r_o[l] = [np.random.uniform(-eps, eps, (lsize, lsize)) for i in range(4)]
            b_z[l], b_i[l], b_f[l], b_o[l] = [np.random.uniform(-eps, eps, (lsize)) for i in range(4)]
        
        # Initialize dense output layer weights
        lsize = layer_sizes[-1]
        input_lsize = layer_sizes[-2]
        eps = np.sqrt(6 / (lsize + input_lsize))
        out_w = np.random.uniform(-eps, eps, (lsize, input_lsize))
        out_b = np.random.uniform(-eps, eps, (lsize))

        # Put parameters in numpy dtype=object array
        self._params = np.array(
            [np.array(w_z, dtype=object), np.array(w_i, dtype=object),
                np.array(w_f, dtype=object), np.array(w_o, dtype=object),
                np.array(r_z, dtype=object), np.array(r_i, dtype=object),
                np.array(r_f, dtype=object), np.array(r_o, dtype=object),
                np.array(b_z, dtype=object), np.array(b_i, dtype=object),
                np.array(b_f, dtype=object), np.array(b_o, dtype=object),
                np.array([out_w], dtype=object),  np.array([out_b], dtype=object)],
            dtype=object
        )  

    @property
    def _mparams(self):
        return self._params
    
    @_mparams.setter
    def _mparams(self, mparams):
        self._params = mparams

    @property
    def _cost_function(self):
        return self._cost_func

    @property
    def _out_size(self):
        return self._lsizes[-1]

    @property
    def nlayers(self):
        '''
        The number of layers in the network.
        '''      
        return self._nlayers

    def _init_gates(self, batch_size):
        self.y[0] = np.zeros((batch_size+2, self._lsizes[0]))

        self.out_ws = np.zeros((batch_size+2, self._lsizes[-1]))
        self.out_a = np.zeros((batch_size+2, self._lsizes[-1]))

        self._bsize = batch_size
        for l in range(1, self._nlayers-1): 
            [self.z_ws[l], self.z[l], self.i_ws[l], self.i[l], self.f_ws[l], 
                self.f[l], self.c[l], self.o_ws[l], self.o[l], 
                self.y[l]] = [np.zeros((batch_size+2, self._lsizes[l])) for i in range(10)]
            
            self.last_y[l] = np.zeros((self._lsizes[l]))
            self.last_c[l] = np.zeros((self._lsizes[l]))

    def _reset_state(self):
        for l in range(1, self._nlayers-1): 
            self.last_y[l] = np.zeros((self._lsizes[l]))
            self.last_c[l] = np.zeros((self._lsizes[l]))

    def reset(self):
        '''
        Resets the hidden state.
        '''
        self.t = 1
        self._reset_state()

    def feed(self, input_data):
        self._outputs = np.zeros((input_data.shape[0], self._out_size))
        if(input_data.ndim == 1):
            # For one input example
            self._feed_ndim1(input_data)
            self._outputs = self.out_a[self.t-1]
        else:
            # For multiple input example
            for i in range(input_data.shape[0]):
                self._feed_ndim1(input_data[i])
                self._outputs[i] = self.out_a[self.t-1]

    def _feed_ndim1(self, input_data):
        if(self.t > self._bsize): self.t = 1
        t = self.t

        self.y[0][t] = input_data
        for l in range(1, self._nlayers-1):
            # Block input
            self.z_ws[l][t] = self.y[l-1][t]@self._params[Wz][l].T + self.last_y[l]@self._params[Rz][l].T + self._params[Bz][l]
            self.z[l][t] = tanh(self.z_ws[l][t])
            # Input gate
            self.i_ws[l][t] = self.y[l-1][t]@self._params[Wi][l].T + self.last_y[l]@self._params[Ri][l].T + self._params[Bi][l]
            self.i[l][t] = sigmoid(self.i_ws[l][t])
            # Forget gate
            self.f_ws[l][t] = self.y[l-1][t]@self._params[Wf][l].T + self.last_y[l]@self._params[Rf][l].T + self._params[Bf][l]
            self.f[l][t] = sigmoid(self.f_ws[l][t])
            # Output gate
            self.o_ws[l][t] = self.y[l-1][t]@self._params[Wo][l].T + self.last_y[l]@self._params[Ro][l].T + self._params[Bo][l]
            self.o[l][t] = sigmoid(self.o_ws[l][t])
            # Cell state
            self.c[l][t] = self.z[l][t]*self.i[l][t] + self.last_c[l]*self.f[l][t]
            # Block output
            self.y[l][t] = tanh(self.c[l][t])*self.o[l][t]

            # Update previous state
            self.last_y[l] = self.y[l][t]
            self.last_c[l] = self.c[l][t]

        # Feed through the last dense layer
        self.out_ws[t] = self.last_y[-1]@self._params[OUT_W][0].T + self._params[OUT_B][0]
        self.out_a[t] = self._output_activ_func(self.out_ws[t])

        self.t += 1

    def get_output(self):
        return self._outputs.squeeze()

    def _init_train(self, batch_size):
        # Initialize LSTM gates
        self._init_gates(batch_size)

    def _backpropagate(self, batch_size, targets):
        # Initialize LSTM gates' gradients
        [dz, di, df, dc, do, dy] = [[np.array([])]*(self._nlayers-1) for i in range(6)]
        for l in range(1, self._nlayers-1): 
            [dz[l], di[l], df[l], dc[l], do[l], dy[l]] = \
                [np.zeros((self._bsize+2, self._lsizes[l])) for i in range(6)]              

        # Initialize LSTM weights' gradients
        [dw_z, dw_i, dw_f, dw_o, dr_z, dr_i, dr_f, dr_o, db_z, 
            db_i, db_f, db_o] = [[np.array([])]*(self._nlayers-1) for i in range(12)]
        for l in range(1, self._nlayers-1):
            lsize = self._lsizes[l]
            input_lsize = self._lsizes[l-1]
            dw_z[l], dw_i[l], dw_f[l], dw_o[l] = [np.zeros((lsize, input_lsize)) for i in range(4)]
            dr_z[l], dr_i[l], dr_f[l], dr_o[l] = [np.zeros((lsize, lsize)) for i in range(4)]
            db_z[l], db_i[l], db_f[l], db_o[l] = [np.zeros((lsize)) for i in range(4)]

        # Initialize dense layer weights' gradients
        lsize = self._lsizes[-1]
        input_lsize = self._lsizes[-2]
        dout_w = np.zeros((lsize, input_lsize))
        dout_b = np.zeros((lsize))
        dcost = np.zeros((lsize))

        def calc_delta_t(l, t):
            if(l == self._nlayers-2):
                return (dcost*dout_a)@self._params[OUT_W][0]
            else:
                return dz[l+1][t]@self._params[Wz][l+1] + di[l+1][t]@self._params[Wi][l+1] \
                    + df[l+1][t]@self._params[Wf][l+1] + do[l+1][t]@self._params[Wo][l+1] 

        for t in range(self._bsize, 0, -1):
            dout_a = self._output_activ_func_prime(self.out_ws[t], self.out_a[t])
            dcost = self._cost_func_prime(self.out_a[t], targets[t-1])
            dout_b += dcost * dout_a
            dout_w += np.multiply.outer(dout_b, self.y[-1][t])

            for l in range(self._nlayers-2, 0, -1):
                dy[l][t] = calc_delta_t(l, t) + dz[l][t+1]@self._params[Rz][l] + di[l][t+1]@self._params[Ri][l]\
                    + df[l][t+1]@self._params[Rf][l] + do[l][t+1]@self._params[Ro][l]
                do[l][t] = dy[l][t] * tanh(self.c[l][t]) * sigmoid_prime(self.o_ws[l][t], self.o[l][t])
                dc[l][t] = dy[l][t] * self.o[l][t] * (1-(tanh(self.c[l][t])**2)) + dc[l][t+1]*self.f[l][t+1]
                df[l][t] = dc[l][t] * self.c[l][t-1] * sigmoid_prime(self.f_ws[l][t], self.f[l][t])
                di[l][t] = dc[l][t] * self.z[l][t] * sigmoid_prime(self.i_ws[l][t], self.i[l][t])
                dz[l][t] = dc[l][t] * self.i[l][t] * tanh_prime(self.z_ws[l][t], self.z[l][t])
                dw_z[l] += np.multiply.outer(dz[l][t], self.y[l-1][t])
                dw_i[l] += np.multiply.outer(di[l][t], self.y[l-1][t])
                dw_f[l] += np.multiply.outer(df[l][t], self.y[l-1][t])
                dw_o[l] += np.multiply.outer(do[l][t], self.y[l-1][t])
                dr_z[l] += np.multiply.outer(dz[l][t+1], self.y[l][t])
                dr_i[l] += np.multiply.outer(di[l][t+1], self.y[l][t])
                dr_f[l] += np.multiply.outer(df[l][t+1], self.y[l][t])
                dr_o[l] += np.multiply.outer(do[l][t+1], self.y[l][t])
                db_z[l] += dz[l][t]
                db_i[l] += di[l][t]
                db_f[l] += df[l][t]
                db_o[l] += do[l][t]

        return np.array(
            [np.array(dw_z, dtype=object), np.array(dw_i, dtype=object),
                np.array(dw_f, dtype=object), np.array(dw_o, dtype=object),
                np.array(dr_z, dtype=object), np.array(dr_i, dtype=object),
                np.array(dr_f, dtype=object), np.array(dr_o, dtype=object),
                np.array(db_z, dtype=object), np.array(db_i, dtype=object),
                np.array(db_f, dtype=object), np.array(db_o, dtype=object),
                np.array([dout_w], dtype=object), np.array([dout_b], dtype=object)
                ],
            dtype=object
        )  

    def _on_test_start(self):
        self._saved_state = (deepcopy(self.last_c), deepcopy(self.last_y))
        self._reset_state()

    def _on_test_end(self):
        self.last_c, self.last_y = self._saved_state

    @property
    def bptt(self):
        return True

    def _get_norm_weights(self):
        return 0

