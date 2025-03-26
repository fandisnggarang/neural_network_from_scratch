import numpy as np
import math

np.random.seed(42)

class Neural_Network(): 
    def __init__(self, input_size, hidden_size, output_size, l_rate, activation='sigmoid', optimizer ='sgd'):
        # initialize data_size, l_rate, and optimizer
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.l_rate       = l_rate
        self.optimizer    = optimizer

        # all results in forward-backward are stored here! 
        self.storage = {}
        self.grads   = {}
        
        # initialize momentum_optimizer
        if self.optimizer == 'momentum': 
            self.momentum_opt = self.initialize_momentum_optimizer()
        
        # initialize activation function
        if activation == 'sigmoid': 
            self.activation = self.sigmoid
        elif activation == 'relu': 
            self.activation = self.relu
        else: 
            raise ValueError("Please choose 'sigmoid' or 'relu' for activation function")
        
        # intialize weight and bias parameter
        self.params = self.initialize()
        
    def apply_dropout(self, x, dropout_prob=0.05):
        # dropout is only used if it is activated and run in training 
        epsilon = 1e-8  # prevent division by zero or small denominators

        self.dropout_mask = np.random.binomial(1, 1-dropout_prob, size = x.shape)

        # make sure that dropout_mask's data type is the same with x's
        self.dropout_mask = self.dropout_mask.astype(x.dtype)
        
        # add epsilon, just in case 1-self.dropout_prob < epsilon, we go with epsilon
        # this is added to avoid denominator being to small.
        x = (x * self.dropout_mask) / max((1 - dropout_prob, epsilon))
        return x
    
    def sigmoid(self, x, derivative=False):
        sig = 1 / (1 + np.exp(-x))
        if derivative:
            return sig * (1 - sig)
        return sig
    
    def relu(self, x, derivative=False): 
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x 
        return np.maximum(0, x)

    def softmax(self, x): 
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis = 0) 

    def initialize(self):
        # if sigmoid, we use Xavier (1.0), else HE (2.0)
        factor = 1.0 if self.activation == self.sigmoid else 2.0
        
        scaling_1 = np.sqrt(factor/self.input_size)
        scaling_2 = np.sqrt(factor/self.hidden_size[0])
        scaling_3 = np.sqrt(factor/self.hidden_size[1])
        scaling_4 = np.sqrt(factor/self.hidden_size[2])

        # create value of each parameter randomly and scale the values using the choosen factor
        params = {
        'w1':np.random.randn(self.hidden_size[0], self.input_size) * scaling_1,
        'b1':np.zeros((self.hidden_size[0], 1)) * scaling_1,

        'w2':np.random.randn(self.hidden_size[1], self.hidden_size[0]) * scaling_2,
        'b2':np.zeros((self.hidden_size[1], 1)) * scaling_2,
        
        'w3':np.random.randn(self.hidden_size[2], self.hidden_size[1]) * scaling_3,
        'b3':np.zeros((self.hidden_size[2], 1)) * scaling_3, 

        'w4':np.random.randn(self.output_size, self.hidden_size[2]) * scaling_4,
        'b4':np.zeros((self.output_size, 1)) * scaling_4   
        }
        
        return params        

    def forward(self, x, training=True, dropout=True):
        self.storage['x'] = x
         
        # The multiplication happens first, and then the neuron is deactivated by dropout
        # input layer -> hidden layer 1
        self.storage['z1'] = np.matmul(self.params['w1'], self.storage['x'].T) + self.params['b1']

        self.storage['a1'] = self.activation(self.storage['z1'], derivative=False)

        # drop some neuron in hidden layer 1
        if training and dropout:
            self.storage['a1'] = self.apply_dropout(self.storage['a1']) 

        # hidden layer 1 -> hidden layer 2
        self.storage['z2'] = np.matmul(self.params['w2'], self.storage['a1']) + self.params['b2']
        self.storage['a2'] = self.activation(self.storage['z2'], derivative=False)

        # drop some neuron in hidden layer 2
        if training and dropout:
            self.storage['a2'] = self.apply_dropout(self.storage['a2']) 

        # hidden layer 2 -> hidden layer 3
        self.storage['z3'] = np.matmul(self.params['w3'], self.storage['a2']) + self.params['b3']
        self.storage['a3'] = self.activation(self.storage['z3'], derivative=False)

        # drop some neuron in hidden layer 3
        if training and dropout:
            self.storage['a3'] = self.apply_dropout(self.storage['a3']) 

        # hidden layer 3 -> output layer
        self.storage['z4']     = np.matmul(self.params['w4'], self.storage['a3']) + self.params['b4']
        self.storage['output'] = self.softmax(self.storage['z4'])

        return self.storage['output']
    
    def cross_entropy_loss(self, output, y, lasso=False, ridge=False, lambda_l=1e-5, lambda_r=1e-5):
        # add clipping to avoid log 0 problems
        epsilon    = 1e-15
        output     = np.clip(output, epsilon, 1.0)
        
        # computing cross-entropy loss
        l_sum = np.sum(np.multiply(y.T, np.log(output)))
        m     = y.shape[0]
        avg_loss = -(1./m) * l_sum 
        
        # if lasso is implemented
        if lasso:
            l_reg  = sum(np.abs(values).sum() for key, values in self.params.items() if 'w' in key)
            avg_loss += lambda_l * l_reg
        
        # if ridge is implemented
        if ridge:
            r_reg  = sum(np.square(values).sum() for key, values in self.params.items() if 'w' in key)
            avg_loss += lambda_r * r_reg
        
        return avg_loss
    
    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis = 1) == np.argmax(output.T, axis = 1))

    def backward(self, output, y, lasso=False, ridge=False, lambda_l=1e-5, lambda_r=1e-5): 
        current_batch_size = y.shape[0]
        output = self.storage['output'].copy()

        # going back: output layer -> hidden layer 3
        dz4 = output - y.T
        self.grads['w4'] = (1./current_batch_size) * (dz4 @ self.storage['a3'].T)
        self.grads['b4'] = (1./current_batch_size) * np.sum(dz4, axis=1, keepdims=True)

        # going back: hidden layer 3 -> hidden layer 2
        da3 = self.params['w4'].T @ dz4 
        dz3 = da3 * self.sigmoid(self.storage['z3'], derivative = True)
        self.grads['w3'] = (1./current_batch_size) * (dz3 @ self.storage['a2'].T)
        self.grads['b3'] = (1./current_batch_size) * np.sum(dz3, axis=1, keepdims=True)

        # going back: hidden layer 2 -> hidden layer 1
        da2 = self.params['w3'].T @ dz3 
        dz2 = da2 * self.sigmoid(self.storage['z2'], derivative = True)
        self.grads['w2'] = (1./current_batch_size) * (dz2 @ self.storage['a1'].T) 
        self.grads['b2'] = (1./current_batch_size) * np.sum(dz2, axis=1, keepdims=True)

        # going back: hidden layer 1 -> input layer
        da1 = self.params['w2'].T @ dz2 
        dz1 = da1 * self.sigmoid(self.storage['z1'], derivative = True)
        self.grads['w1'] = (1./current_batch_size) * (dz1 @ self.storage['x'])
        self.grads['b1'] = (1./current_batch_size) * np.sum(dz1, axis=1, keepdims=True)
        
        # add lasso and ridge penalty to the weights
        for key in [k for k in self.grads if 'w' in k]: 
            if lasso: 
                self.grads[key] += lambda_l * np.sign(self.params[key])
            if ridge: 
                self.grads[key] += 2 * lambda_r * self.params[key]
                    
        return self.grads 
                        
    def apply_gradient_clipping(self, clip_value):
        # apply clipping to avoid gradient exploding
        for key in [k for k in self.grads if 'w' in k]: 
            grad_norm = np.linalg.norm(self.grads[key])
            if grad_norm > clip_value: 
                    self.grads[key] *= (clip_value/grad_norm)
        return self.grads

    def initialize_momentum_optimizer(self):
        # initialize momentum 
        momentum_opt = {
            'w1': np.zeros_like(self.params['w1']),
            'b1': np.zeros_like(self.params['b1']),
            'w2': np.zeros_like(self.params['w2']),
            'b2': np.zeros_like(self.params['b2']), 
            'w3': np.zeros_like(self.params['w3']),
            'b3': np.zeros_like(self.params['b3']),
            'w4': np.zeros_like(self.params['w4']),
            'b4': np.zeros_like(self.params['b4']), 
        }
        return momentum_opt 

    def optimize(self, beta_momentum=0.9):
        # if sgd technique is used
        if self.optimizer == 'sgd':
            for key in self.params:
                self.params[key] -= self.l_rate * self.grads[key]

        elif self.optimizer == 'momentum':
        # if momentum technique is used
            for key in self.params:
                self.momentum_opt[key] = (beta_momentum  * self.momentum_opt[key] + (1. - beta_momentum) * self.grads[key])
                self.params[key]      -= self.l_rate * self.momentum_opt[key]
        else:
            raise ValueError("We only have 'sgd' and 'momentum'. Please choose one!")
        
        return self.params
    
# the codes above was modified from https://github.com/lionelmessi6410/Neural-Networks-from-Scratch/blob/main/model.py
