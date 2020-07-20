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

    def __init__(self, parameters):
        self.model = tf.keras.Sequential()
        self.train_optimiser = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.opt_optimiser = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.parameters = parameters

    def generate_training_input(self):
        pass

    def construct_network(self):
        pass

    def generate_training_output(self, output):
        pass

    def train(self):
        pass

    def save_training_data(self, filename):
        pass

    def find_optimal_params(self):
        pass

    def _normalise(self, x):
        pass
    
    def _normalise_array(self, x):
        pass
    
    def _inverse_normalise(self, x_norm):
        pass
    
    def _inverse_normalise_array(self, x_norm):
        pass
    
    def plot_history(self, filename):
        pass

    def plot_network(self, filename):
        pass
    
    def predict(self, x):
        pass