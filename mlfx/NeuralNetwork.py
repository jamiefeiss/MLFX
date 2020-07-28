import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # mute warnings, etc.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .variables import *

class NeuralNetwork(object):
    """
    Class for the neural network
    """

    """
    Attributes:
        model
        optimisers
        settings
        training data
        parameters
        history
        unique simulation ID
    """

    """
    Methods:
        generate random training data points
        generate training output
        normalise data point
        normalise list
        un-normalise data point
        un-normalise list
        save training data to file
        checkpoint/save model
        load model
        train
        get optimal parameters
        refine
        add training point
        train from one data point
        plot history
        plot network structure
    """

    def __init__(self, parameters, settings):
        self.parameters = parameters
        self.settings = settings
        self.model = tf.keras.Sequential()
        self.train_optimiser = tf.keras.optimizers.Adam(learning_rate=settings['train_learning_rate'] or 0.01)
        self.opt_optimiser = tf.keras.optimizers.Adam(learning_rate=settings['opt_learning_rate'] or 0.01)

    def generate_training_input(self):
        inputs = []
        inputs_normalised = []
        for i in range(len(self.parameters)):
            param = [param for param in self.parameters if param.index == i]
            param_list = np.random.uniform(low = param.min, high = param.max, size = self.settings['training_size']) # random sampling
            inputs.append(param_list)
            inputs_normalised.append(self._normalise_array(param_list, param.min, param.max))

        self.training_input = np.column_stack(tuple([input for input in inputs]))
        self.training_input_norm = np.column_stack(tuple([input for input in inputs_normalised]))    

    def generate_training_output(self, objective_fn):
        y_list = [objective_fn(x) for x in self.training_input] # calculate output from objective function
        self.training_output = np.asarray(y_list, dtype = np.float64).reshape(self.settings['training_size'], 1)
        self.training_output_norm = np.asarray(self._normalise_array(y_list, min(y_list),max(y_list)), dtype = np.float64).reshape(self.settings['training_size'], 1)
    
    def construct_network(self):
        for i in range(len(self.settings['neurons'])):
            if i == 0:
                self.model.add(tf.keras.layers.Dense(self.settings['neurons'][i], input_shape=[len(self.parameters)], activation='sigmoid')) # first layer
            else:
                self.model.add(tf.keras.layers.Dense(self.settings['neurons'][i], activation='sigmoid'))
        self.model.add(tf.keras.layers.Dense(1)) # output layer
        self.model.compile(optimizer=self.train_optimiser, loss='mse')

    def train(self):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.settings['early_stop_patience'], min_delta=self.settings['early_stop_delta'])
        self.history = self.model.fit(self.training_input_norm, self.training_output_norm,
                                epochs=self.settings['train_epochs'], validation_split = self.settings['validation_split'],
                                callbacks=[early_stop]
                            )

    def save_training_data(self, filename):
        pass
    
    def cost(self, y_norm, cost_fn):
        y_min = min(self.training_output)
        y_max = max(self.training_output)
        y = self._inverse_normalise(y_norm, y_min, y_max)
        y_cost = cost_fn(y)
        return self._normalise(y_cost, y_min, y_max))

    def find_optimal_params(self):
        default_param_values_norm = [] # normalised default parameters
        for param in self.parameters:
            default_param_values_norm.append(self._normalise(param.value, param.min, param.max))
        inputs_default = tf.constant(default_param_values_norm)
        inputs_opt = tf.Variable([inputs_default], trainable=True, dtype = tf.float32)

        for i in range(self.settings['opt_epochs']):
            with tf.GradientTape() as tape:
                tape.watch(inputs_opt) 
                C = self.model(inputs_opt)[0]
            g = tape.gradient(C, [inputs_opt])
            self.opt_optimiser.apply_gradients(zip(g, [inputs_opt]))
        
        self.optimal_parameters = inputs_opt

    def _normalise(x, min, max):
        return (x - min) / (max - min)
    
    def _normalise_array(x_list, min, max):
        return [normalise(x, min, max) for x in x_list]
    
    def _inverse_normalise(x_norm, min, max):
        return x_norm * (max - min) + min
    
    def _inverse_normalise_array(x_norm_list, min, max):
        return [inverse_normalise(x_norm, min, max) for x_norm in x_norm_list]
    
    def plot_history(self, filename):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = np.arange(1, len(loss) + 1)

        # plot
        fig, ax = plt.subplots()
        ax.plot(epochs, loss, label='Training')
        ax.plot(epochs, val_loss, label='Validation')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.set_ylim([0, None])
        ax.legend()
        fig.savefig(filename + '.png')

    def plot_network(self, filename):
        pass
    
    def predict(self, x):
        return self.model(x)[0]