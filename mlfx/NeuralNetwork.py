import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # mute warnings, etc.
import time
from typing import Dict, List, Optional, Union, Type, Callable

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
        parameters (List[Parameter]): The list of input parameters to be trained
        settings (Dict): The ML settings

    Attributes:
        parameters (List[Parameter]): The list of input parameters to be trained
        settings (Dict): A dict of the machine learning-specific settings
        model (tf.keras.Sequential): The neural network model
        train_optimiser (tf.keras.optimizers.Adam): The optimiser for training the model
        opt_optimiser (tf.keras.optimizers.Adam): The optimiser for optimising the model
        training_input (numpy.ndarray): An array of the training data input values
        training_input_norm (numpy.ndarray): An array of the nromalised training data input values
        training_output (numpy.ndarray): An array of the training data output values
        training_output_norm (numpy.ndarray): An array of the normalised training data output values
        history (tf.keras.callbacks.History): A history object that records training error over time
    """

    def __init__(self, parameters: List[Type[Parameter]], settings: Dict):
        self.parameters = parameters
        self.settings = settings
        self.model = tf.keras.Sequential()
        if settings['train_learning_decay']:
            train_learning_decay = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=settings['train_learning_rate'], decay_steps=1.0, decay_rate=0.5)
            self.train_optimiser = tf.keras.optimizers.Adam(learning_rate=train_learning_decay)
        else:
            self.train_optimiser = tf.keras.optimizers.Adam(learning_rate=settings['train_learning_rate'])
        if settings['opt_learning_decay']:
            opt_learning_decay = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=settings['opt_learning_rate'], decay_steps=1.0, decay_rate=0.5)
            self.opt_optimiser = tf.keras.optimizers.Adam(learning_rate=opt_learning_decay)
        else:
            self.opt_optimiser = tf.keras.optimizers.Adam(learning_rate=settings['opt_learning_rate'])

    def _input_data_row_string(self, row: Union[List, Type[numpy.ndarray]]):
        """
        Formats input data to be written to file

        Args:
            row (List[int, float], numpy.ndarray):

        Returns:
            str: The formatted string to be written to file
        """
        string = ''
        for i, element in enumerate(row):
            if i == len(row) - 1: # end of row
                string += str(row[i]) + '\n'
            else:
                string += str(row[i]) + ' '
        return string

    def generate_training_input(self):
        """
        Generates a training set for the input data
        
        A training set of the specified size is generated from either: linearly spaced, uniform random or normal random
        distributed within the ranges of each parameter. This function also generated a histogram for each parameter, showing
        the values chosen. A normalised set of training is also made in this function. After the training set has been generated,
        the input values are written to file.
        """
        inputs = []
        inputs_normalised = []
        for i in range(len(self.parameters)):
            param = [param for param in self.parameters if param.index == i][0] # enforce parameter order by index
            param_list = np.random.uniform(low = param.min, high = param.max, size = self.settings['training_size']) # random sampling
            # param_list = np.linspace(start = param.min, stop = param.max, num = self.settings['training_size']) # evenly spaced sampling
            # param_list = np.random.normal(loc = param.value, scale = 0.3, size = self.settings['training_size'])
            inputs.append(param_list)
            inputs_normalised.append(self._normalise_array(param_list, param.min, param.max))

            self.plot_param_hist(param, param_list)

        self.training_input = np.column_stack(tuple([input for input in inputs]))
        self.training_input_norm = np.column_stack(tuple([input for input in inputs_normalised]))

        # clear file
        file = open("input.txt","w")
        file.close()

        # write training input data to file
        with open('input.txt', 'w') as f:
            for row in self.training_input:
                f.write(self._input_data_row_string(row))

    @timer
    def generate_training_output(self, objective_fn: Callable):
        """
        Generates the output training data from the input parameter values.

        Args:
            objective_fn (function): The objective function that runs XMDS2, returning a cost value
        """
        print('Generating training data...')

        # clear file
        file = open("output.txt","w")
        file.close()

        y_list = []
        counter = 0
        print('...{}/{} training points completed...'.format(counter, self.settings['training_size']), end="\r", flush=True)
        for x in self.training_input:
            y = objective_fn(x) # run xmds
            counter += 1
            if counter == self.settings['training_size']:
                print('{}/{} training points completed.      '.format(counter, self.settings['training_size']))
            else:
                print('...{}/{} training points completed...'.format(counter, self.settings['training_size']), end="\r", flush=True)

            # append to output file
            with open('output.txt', 'a') as f:
                f.write(str(y) + '\n')
            y_list.append(y)
        
        self.training_output = np.asarray(y_list, dtype = np.float64).reshape(self.settings['training_size'], 1)
        self.training_output_norm = np.asarray(self._normalise_array(y_list, min(y_list),max(y_list)), dtype = np.float64).reshape(self.settings['training_size'], 1)

        # self.plot_param_scatter(self.training_input, self.training_output)

    def construct_network(self):
        """Initialises the neural network"""
        # add neuron layers
        for i, layer_count in enumerate(self.settings['neurons']):
            if i == 0:
                self.model.add(tf.keras.layers.Dense(layer_count, input_shape=[len(self.parameters)], activation='sigmoid')) # first layer
            else:
                self.model.add(tf.keras.layers.Dense(layer_count, activation='sigmoid'))
        self.model.add(tf.keras.layers.Dense(1)) # output layer
        self.model.compile(optimizer=self.train_optimiser, loss='mse')
        self.model.summary()

    @timer
    def train(self):
        """Trains the neural network on the training set"""
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

    def _write_optimal_to_file(self, optimal: Type[numpy.ndarray]):
        """
        Writes the optimal parameter attempts to file
        
        Args:
            optimal (numpy.ndarray): the current optimal set attempt
        """
        optimal_list = self._inverse_normalise_parameter_row(optimal)
        optimal_list_formatted = [param[0] for param in optimal_list]
        with open('optimal.txt', 'a') as f:
            f.write(self._input_data_row_string(optimal_list_formatted))
    
    def _write_cost_to_file(self, cost: Union[int, float]):
        """
        Writes the cost value of the current attempt to file
        
        Args:
            cost (int, float): the current cost value
        """
        cost_string = str(cost) + '\n'
        with open('cost.txt', 'a') as f:
            f.write(cost_string)

    # @timer
    # def find_optimal_params3(self, objective_fn):
    #     """Runs an optimiser around each optimal guess"""
    #     print('Optimising...')
    #     # initialise to default parameter values
    #     default_param_values_norm = []
    #     for param in self.parameters:
    #         default_param_values_norm.append(self._normalise(param.value, param.min, param.max))
    #     inputs_default = tf.constant(default_param_values_norm)
    #     inputs_refine = tf.Variable([inputs_default], trainable=True, dtype = tf.float32)
    #     refine_optimiser = tf.keras.optimizers.Adam(learning_rate=self.settings['refine_learning_rate'])

    #     def opt_fn(var):
    #         # Can't find connection between variable and cost to calculate gradient
    #         # TODO try tf.nn.softmax... normalisation throughout instead of minmax -> (https://github.com/tensorflow/tensorflow/issues/1511)
    #         x_unnorm = self._inverse_normalise_parameter_row(var.numpy())
    #         print('x={}'.format(x_unnorm))
    #         y_unnorm = objective_fn(x_unnorm[0]) # run xmds - passing array, need [0] to be a float
    #         print('y={}'.format(y_unnorm))
    #         C = self._normalise(y_unnorm, self.training_output.min(), self.training_output.max())
    #         return tf.constant([C]) # cost needs to be a tensor

    #     # refine
    #     for i in range(self.settings['refine_epochs']):
    #         inputs_opt = tf.Variable(inputs_refine, trainable=True, dtype = tf.float32)
    #         opt_optimiser = tf.keras.optimizers.Adam(learning_rate=self.settings['opt_learning_rate'])

    #         # optimise
    #         for i in range(self.settings['opt_epochs']):
    #             with tf.GradientTape() as tape_opt:
    #                 tape_opt.watch(inputs_opt)
    #                 C_opt = self.predict(inputs_opt)
    #                 print(C_opt)
    #                 print(inputs_opt)
    #             g_opt = tape_opt.gradient(C_opt, [inputs_opt])
    #             opt_optimiser.apply_gradients(zip(g_opt, [inputs_opt]))
            
    #         inputs_refine.assign(inputs_opt)

    #         with tf.GradientTape() as tape_refine:
    #             tape_refine.watch(inputs_refine)
    #             C_refine = opt_fn(inputs_refine)
    #             print(C_refine)
    #             print(inputs_refine)
    #         g_refine = tape_refine.gradient(C_refine, [inputs_refine])
    #         refine_optimiser.apply_gradients(zip(g_refine, [inputs_refine]))

    #         y_output = np.asarray(C, dtype = np.float32).reshape(1, 1)

    #         self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    #         self.model.fit(x_norm, y_output, epochs=10, verbose=0)

    # @timer
    # def find_optimal_params2(self, objective_fn):
    #     """Trains the model after each optimisation run"""
    #     print('Optimising...')

    #     # # save to file
    #     # self._write_optimal_to_file(inputs_opt.numpy()) # default initial values

    #     # refine
    #     for i in range(self.settings['refine_epochs']):
    #         default_param_values_norm = [] # normalised default parameters
    #         for param in self.parameters:
    #             default_param_values_norm.append(self._normalise(param.value, param.min, param.max))
    #         inputs_default = tf.constant(default_param_values_norm)
    #         inputs_opt = tf.Variable([inputs_default], trainable=True, dtype = tf.float32)

    #         opt_optimiser = tf.keras.optimizers.Adam(learning_rate=self.settings['opt_learning_rate'])

    #         # save to file
    #         # self._write_optimal_to_file(inputs_opt.numpy()) # default initial values
    #         for i in range(self.settings['opt_epochs']):
    #             with tf.GradientTape() as tape:
    #                 tape.watch(inputs_opt)
    #                 C = self.predict(inputs_opt)
    #             g = tape.gradient(C, [inputs_opt])
    #             opt_optimiser.apply_gradients(zip(g, [inputs_opt]))

    #             # save to file
    #             # self._write_optimal_to_file(inputs_opt.numpy())
            
    #         self.optimal_parameters = inputs_opt.numpy()

    #         x_unnorm = self._inverse_normalise_parameter_row(self.optimal_parameters)
    #         y_unnorm = objective_fn(x_unnorm[0]) # run xmds - passing array, need [0] to be a float
    #         y_norm = self._normalise(y_unnorm, self.training_output.min(), self.training_output.max())
    #         y_output = np.asarray(y_norm, dtype = np.float32).reshape(1, 1)

    #         self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    #         self.model.fit(self.optimal_parameters, y_output, epochs=10, verbose=0)
        
    #     # find final optimal params
    #     default_param_values_norm = [] # normalised default parameters
    #     for param in self.parameters:
    #         default_param_values_norm.append(self._normalise(param.value, param.min, param.max))
    #     inputs_default = tf.constant(default_param_values_norm)
    #     inputs_opt = tf.Variable([inputs_default], trainable=True, dtype = tf.float32)

    #     final_optimiser = tf.keras.optimizers.Adam(learning_rate=self.settings['opt_learning_rate'])

    #     # save to file
    #     # self._write_optimal_to_file(inputs_opt.numpy()) # default initial values

    #     for i in range(self.settings['opt_epochs']):
    #         with tf.GradientTape() as tape:
    #             tape.watch(inputs_opt)
    #             C = self.predict(inputs_opt)
    #         g = tape.gradient(C, [inputs_opt])
    #         final_optimiser.apply_gradients(zip(g, [inputs_opt]))

    #         # save to file
    #         # self._write_optimal_to_file(inputs_opt.numpy())
        
    #     self.optimal_parameters = inputs_opt.numpy()

    #     x_unnorm = self._inverse_normalise_parameter_row(self.optimal_parameters)
    #     print('x={}'.format(x_unnorm))
    #     y_unnorm = objective_fn(x_unnorm[0]) # run xmds - passing array, need [0] to be a float
    #     print('y={}'.format(y_unnorm))
    #     y_norm = self._normalise(y_unnorm, self.training_output.min(), self.training_output.max())
    #     y_output = np.asarray(y_norm, dtype = np.float32).reshape(1, 1)

    @timer
    def find_optimal_params(self, objective_fn: Callable):
        """
        Does a single optimisation run over the parameter surface
        
        Args:
            objective_fn (function): The objective function that runs XMDS2, returning a cost value
        """
        print('Optimising...')

        # clear file
        file = open("optimal.txt","w")
        file.close()
        file = open("cost.txt","w")
        file.close()

        default_param_values_norm = [] # normalised default parameters
        for param in self.parameters:
            default_param_values_norm.append(self._normalise(param.value, param.min, param.max))
        inputs_default = tf.constant(default_param_values_norm)
        inputs_opt = tf.Variable([inputs_default], trainable=True, dtype = tf.float32)

        opt_optimiser = self.opt_optimiser

        # save to file
        self._write_optimal_to_file(inputs_opt.numpy()) # default initial values
        for i in range(self.settings['opt_epochs']):
            with tf.GradientTape() as tape:
                tape.watch(inputs_opt)
                C = self.predict(inputs_opt)
            g = tape.gradient(C, [inputs_opt])
            opt_optimiser.apply_gradients(zip(g, [inputs_opt]))

            # save to file
            self._write_optimal_to_file(inputs_opt.numpy())
        
        self.optimal_parameters = inputs_opt.numpy()

        x_unnorm = self._inverse_normalise_parameter_row(self.optimal_parameters)
        x_final = []
        for i in range(len(x_unnorm)):
            x_final.append(x_unnorm[i][0])
        print('x={}'.format(x_final))
        y_unnorm = objective_fn(x_final) # run xmds - passing array, need [0] to be a float
        self._write_cost_to_file(y_unnorm)
        print('y={}'.format(y_unnorm))
        y_norm = self._normalise(y_unnorm, self.training_output.min(), self.training_output.max())
        y_output = np.asarray(y_norm, dtype = np.float32).reshape(1, 1)

    def _normalise(self, x: Union[int, float], min: Union[int, float], max: Union[int, float]) -> Union[int, float]:
        """
        Normalises a number using min-max normalisation
        
        Args:
            x (int, float): The number to be normalised
            min (int, float): The minimum value
            max (int, float): The maximum value
        
        Returns:
            int, float: The normalised value
        """
        return (x - min) / (max - min)
    
    def _normalise_array(self, x_list: Union[List[Union[int, float]], Type[numpy.ndarray]], min: Union[int, float], max: Union[int, float]) -> List[Union[int, float]]:
        """
        Normalises an array of numbers using min-max normalisation
        
        Args:
            x_list (List[int, float], numpy.ndarray): The array to be normalised
            min (int, float): The minimum value
            max (int, float): The maximum value
        
        Returns:
            List[int, float]: The normalised array
        """
        return [self._normalise(x, min, max) for x in x_list]
    
    def _inverse_normalise(self, x_norm: Union[int, float], min: Union[int, float], max: Union[int, float]) -> Union[int, float]:
        """
        Inverse-normalises a number using min-max normalisation
        
        Args:
            x_norm (int, float): The normalised number
            min (int, float): The minimum value
            max (int, float): The maximum value
        
        Returns:
            int, float: The inverse-normalised value
        """
        return x_norm * (max - min) + min
    
    def _inverse_normalise_array(self, x_norm_list: Union[List[Union[int, float]], Type[numpy.ndarray]], min: Union[int, float], max: Union[int, float]) -> List[Union[int, float]]:
        """
        Inverse-normalises an array of numbers using min-max normalisation
        
        Args:
            x_norm_list (List[int, float], numpy.ndarray): The normalised array
            min (int, float): The minimum value
            max (int, float): The maximum value
        
        Returns:
            List[int, float]: The inverse-normalised array
        """
        return [self._inverse_normalise(x_norm, min, max) for x_norm in x_norm_list]
    
    def _inverse_normalise_parameter_row(self, x_norm: Type[numpy.ndarray]) -> Type[numpy.ndarray]:
        """
        Inverse-normalises a row of parameter values using min-max normalisation
        
        Args:
            x_norm (numpy.ndarray): The row of normalised parameter values
        
        Returns:
            numpy.ndarray: The row of inverse-normalised parameter values
        """
        x_list = []
        for i, param in enumerate(x_norm[0]):
            x_list.append(self._inverse_normalise(param, self.parameters[i].min, self.parameters[i].max))
        return np.asarray(x_list, dtype = np.float64).reshape(len(self.parameters), 1)
    
    def plot_history(self):
        """Plots the training history"""
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
        ax.set_title('Model Loss')
        ax.legend()
        fig.savefig('history.png')
    
    def plot_param_hist(self, param: Type[Parameter], param_list: Type[numpy.ndarray]):
        """
        Plots a histogram of the samples values for a parameter
        
        Args:
            param (Parameter): The parameter to be plotted
            param_list (List[int, float]): The array of parameter values
        """
        fig, ax = plt.subplots()
        ax.hist(param_list, bins=10, edgecolor='black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Random sample of {}'.format(param.name))
        fig.savefig(param.name + '_input.png')
    
    def plot_param_scatter(self, param_list, output_list):
        """
        Generates a scatter plot for the parameter values and corresponding cost values
        
        Args:
            arg1 (type): desc
            arg1 (type): desc
        """
        fig, ax = plt.subplots()
        ax.scatter(param_list, output_list)
        fig.savefig('training.png')

    def plot_network(self):
        pass
    
    def predict(self, x: Type[tf.Variable]) -> Union[int, float]:
        """
        Predicts the cost value for a set of input values

        This function also writes the predicted cost to file.
        
        Args:
            x (tf.Variable): The input values
        
        Returns:
            int, float: The predicted cost value
        """
        y = self.model(x)[0]

        y_unnorm = self._inverse_normalise(y.numpy()[0], self.training_output.min(), self.training_output.max())
        self._write_cost_to_file(y_unnorm)
        
        # constrain variable to parameter ranges
        # for feature in x.numpy():
        #     if feature < 0: # below min
        #         y += np.exp(-100 * feature) - 1
        #     elif feature > 1: # above max
        #         y += np.exp(100 * feature) - 1
        return y