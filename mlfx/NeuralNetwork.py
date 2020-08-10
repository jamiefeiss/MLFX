import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # mute warnings, etc.
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .variables import *

def timer(func):
    """A timer decorator"""
    def function_timer(*args, **kwargs):
        """A nested function for timing other functions"""
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "The runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__, time=round(runtime, 4)))
        return value
    return function_timer

class NeuralNetwork(object):
    """
    Class for the neural network

    Args:
        parameters
        settings

    Attributes:
        parameters
        settings
        model
        train_optimiser
        opt_optimiser
        training_input
        training_input_norm
        training_output
        training_output_norm
        history
    """

    def __init__(self, parameters, settings):
        self.parameters = parameters
        self.settings = settings
        self.model = tf.keras.Sequential()
        self.train_optimiser = tf.keras.optimizers.Adam(learning_rate=settings['train_learning_rate'] or 0.01)
        self.opt_optimiser = tf.keras.optimizers.Adam(learning_rate=settings['opt_learning_rate'] or 0.01)

    def _input_data_row_string(self, row):
        string = ''
        for i, element in enumerate(row):
            if i == len(row) - 1: # end of row
                string += str(row[i]) + '\n'
            else:
                string += str(row[i]) + ','
        return string

    def generate_training_input(self):
        inputs = []
        inputs_normalised = []
        for i in range(len(self.parameters)):
            param = [param for param in self.parameters if param.index == i][0] # enforce parameter order by index
            param_list = np.random.uniform(low = param.min, high = param.max, size = self.settings['training_size']) # random sampling
            # param_list = np.linspace(start = param.min, stop = param.max, num = self.settings['training_size']) # evenly spaced sampling
            inputs.append(param_list)
            inputs_normalised.append(self._normalise_array(param_list, param.min, param.max))

        self.training_input = np.column_stack(tuple([input for input in inputs]))
        self.training_input_norm = np.column_stack(tuple([input for input in inputs_normalised]))

        # write training input data to file
        # with open('input.txt', 'w') as f:
        #     for row in self.training_input:
        #         f.write(self._input_data_row_string(row))

    @timer
    def generate_training_output(self, objective_fn):
        print('Generating training data...')
        # self.objective_fn = objective_fn
        y_list = []
        for x in self.training_input:
            y = objective_fn(x) # run xmds

            # append to output file
            # with open('output.txt', 'a') as f:
            #     f.write(str(y) + '\n')
            y_list.append(y)
        
        self.training_output = np.asarray(y_list, dtype = np.float64).reshape(self.settings['training_size'], 1)
        self.training_output_norm = np.asarray(self._normalise_array(y_list, min(y_list),max(y_list)), dtype = np.float64).reshape(self.settings['training_size'], 1)
    
    def construct_network(self):
        # add neuron layers
        for i, layer_count in enumerate(self.settings['neurons']):
            if i == 0:
                self.model.add(tf.keras.layers.Dense(layer_count, input_shape=[len(self.parameters)], activation='sigmoid')) # first layer
            else:
                self.model.add(tf.keras.layers.Dense(layer_count, activation='sigmoid'))
        self.model.add(tf.keras.layers.Dense(1)) # output layer
        
        self.model.compile(optimizer=self.train_optimiser, loss='mse')

    @timer
    def train(self):
        print('Training network...')
        if self.settings['early_stop']:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.settings['early_stop_patience'], min_delta=self.settings['early_stop_delta'])
            self.history = self.model.fit(self.training_input_norm, self.training_output_norm,
                                    epochs=self.settings['train_epochs'], validation_split = self.settings['validation_split'],
                                    callbacks=[early_stop], verbose=0
                                )
        else:
            self.history = self.model.fit(self.training_input_norm, self.training_output_norm,
                                    epochs=self.settings['train_epochs'], validation_split = self.settings['validation_split'],
                                    verbose=0
                                )
        # save model
        # self.model.save('saved_model')

    # def _write_optimal_to_file(self, optimal):
    #     optimal_list = []
    #     for i, input in enumerate(optimal):
    #         optimal_list.append(self._inverse_normalise(input, self.parameters[i].min, self.parameters[i].max))

    #     with open('optimal.txt', 'a') as f:
    #         f.write(self._input_data_row_string(optimal_list))

    @timer
    def find_optimal_params(self, objective_fn):
        print('Optimising...')

        # default_param_values_norm = [] # normalised default parameters
        # for param in self.parameters:
        #     default_param_values_norm.append(self._normalise(param.value, param.min, param.max))
        # inputs_default = tf.constant(default_param_values_norm)
        # # inputs_opt = tf.Variable([inputs_default], trainable=True, dtype = tf.float32, constraint=tf.keras.constraints.MinMaxNorm()) # constraint
        # inputs_opt = tf.Variable([inputs_default], trainable=True, dtype = tf.float32)

        # # save to file
        # self._write_optimal_to_file(inputs_opt.numpy()) # default initial values

        # refine
        for i in range(self.settings['refine_epochs']):
            default_param_values_norm = [] # normalised default parameters
            for param in self.parameters:
                default_param_values_norm.append(self._normalise(param.value, param.min, param.max))
            inputs_default = tf.constant(default_param_values_norm)
            # inputs_opt = tf.Variable([inputs_default], trainable=True, dtype = tf.float32, constraint=tf.keras.constraints.MinMaxNorm()) # constraint
            inputs_opt = tf.Variable([inputs_default], trainable=True, dtype = tf.float32)

            opt_optimiser = tf.keras.optimizers.Adam(learning_rate=self.settings['opt_learning_rate'])

            # save to file
            # self._write_optimal_to_file(inputs_opt.numpy()) # default initial values
            for i in range(self.settings['opt_epochs']):
                with tf.GradientTape() as tape:
                    tape.watch(inputs_opt)
                    C = self.predict(inputs_opt)
                g = tape.gradient(C, [inputs_opt])
                opt_optimiser.apply_gradients(zip(g, [inputs_opt]))

                # save to file
                # self._write_optimal_to_file(inputs_opt.numpy())
            
            self.optimal_parameters = inputs_opt.numpy()
            # print('opt_params={}'.format(self.optimal_parameters))

            x_unnorm = self._inverse_normalise_parameter_row(self.optimal_parameters)
            print('x={}'.format(x_unnorm))
            y_unnorm = objective_fn(x_unnorm[0]) # run xmds - passing array, need [0] to be a float
            print('y={}'.format(y_unnorm))
            y_norm = self._normalise(y_unnorm, self.training_output.min(), self.training_output.max())
            y_output = np.asarray(y_norm, dtype = np.float32).reshape(1, 1)

            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
            self.model.fit(self.optimal_parameters, y_output, epochs=10, verbose=0)
        
        # find final optimal params
        default_param_values_norm = [] # normalised default parameters
        for param in self.parameters:
            default_param_values_norm.append(self._normalise(param.value, param.min, param.max))
        inputs_default = tf.constant(default_param_values_norm)
        # inputs_opt = tf.Variable([inputs_default], trainable=True, dtype = tf.float32, constraint=tf.keras.constraints.MinMaxNorm()) # constraint
        inputs_opt = tf.Variable([inputs_default], trainable=True, dtype = tf.float32)

        final_optimiser = tf.keras.optimizers.Adam(learning_rate=self.settings['opt_learning_rate'])

        # save to file
        # self._write_optimal_to_file(inputs_opt.numpy()) # default initial values

        for i in range(self.settings['opt_epochs']):
            with tf.GradientTape() as tape:
                tape.watch(inputs_opt)
                C = self.predict(inputs_opt)
            g = tape.gradient(C, [inputs_opt])
            final_optimiser.apply_gradients(zip(g, [inputs_opt]))

            # save to file
            # self._write_optimal_to_file(inputs_opt.numpy())
        
        self.optimal_parameters = inputs_opt.numpy()
        # print('opt_params={}'.format(self.optimal_parameters))

        x_unnorm = self._inverse_normalise_parameter_row(self.optimal_parameters)
        print('x={}'.format(x_unnorm))
        y_unnorm = objective_fn(x_unnorm[0]) # run xmds - passing array, need [0] to be a float
        print('y={}'.format(y_unnorm))
        y_norm = self._normalise(y_unnorm, self.training_output.min(), self.training_output.max())
        y_output = np.asarray(y_norm, dtype = np.float32).reshape(1, 1)
    
    def _inverse_normalise_parameter_row(self, x_norm):
        x_list = []
        for i, param in enumerate(x_norm):
            x_list.append(self._inverse_normalise(param, self.parameters[i].min, self.parameters[i].max))
        return np.asarray(x_list, dtype = np.float64).reshape(len(self.parameters), 1)

    def _normalise(self, x, min, max):
        return (x - min) / (max - min)
    
    def _normalise_array(self, x_list, min, max):
        return [self._normalise(x, min, max) for x in x_list]
    
    def _inverse_normalise(self, x_norm, min, max):
        return x_norm * (max - min) + min
    
    def _inverse_normalise_array(self, x_norm_list, min, max):
        return [self._inverse_normalise(x_norm, min, max) for x_norm in x_norm_list]
    
    def plot_history(self):
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
        fig.savefig('history.png')

    def plot_network(self):
        pass
    
    def predict(self, x):
        # print('x: {}'.format(x.numpy()))
        y = self.model(x)[0]

        # constrain variable to parameter ranges
        # for feature in x.numpy():
        #     if feature < 0: # below min
        #         y += np.exp(-100 * feature) - 1
        #     elif feature > 1: # above max
        #         y += np.exp(100 * feature) - 1
        return y