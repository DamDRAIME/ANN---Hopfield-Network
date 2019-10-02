import numpy as np
from math import log, sqrt
from bokeh.io import show, output_notebook, output_file
from bokeh.palettes import RdBu3
from bokeh.layouts import gridplot, grid
from bokeh.plotting import figure


class HopfieldNet(object):
    def __init__(self):
        self.patterns = None
        self.weights = None
        self.learning_rule = None
        self.nbr_attractors = None
        self.nbr_neurons = None
        self.original_shape = None
        self._capacity = None
        self.threshold = None

    def store(self, data: (list, np.ndarray), learning_rule: str = 'Hebbian'):
        """
        Function used to store some patterns in the hopfield network. Those patterns should be provided in a list,
        called data, where each element is a pattern to be saved. The learning_rule parameters can be changed to apply
        the Storkey rule instead of the default Hebbian rule.

        :param data: list of patterns to be stored. Those can be 1D or 2D arrays. Patterns must be made of (1 and 0),
        or (1 and -1)
        :param learning_rule: 'Hebbian' or 'Storkey'
        :return: None
        """

        assert self.weights is None, 'You have already stored some patterns'
        assert learning_rule in ['Hebbian', 'Storkey'], 'The learning_rule should be Hebbian (default) or Storkey'

        self.original_shape, self.patterns = self._convert(data)
        self.learning_rule = learning_rule
        self.nbr_attractors, self.nbr_neurons = self.patterns.shape

        weights = np.zeros((self.nbr_neurons, self.nbr_neurons))

        if self.learning_rule == 'Hebbian':
            for pattern in self.patterns:
                weights += np.outer(pattern, pattern)
            np.fill_diagonal(weights, 0)
            self.weights = weights / self.nbr_attractors

        elif self.learning_rule == 'Storkey':
            for pattern in self.patterns:
                hij = np.array([np.sum(weights * pattern, axis=1), ] * self.nbr_neurons).transpose()
                cij = weights * pattern
                hij -= cij
                hij -= np.diag(cij).reshape((-1, 1))
                hij += np.diag(np.diag(cij))
                eiej = np.outer(pattern, pattern)
                hijej = pattern * hij
                eihji = hijej.transpose()
                weights += (eiej - eihji - hijej) / self.nbr_neurons
            np.fill_diagonal(weights, 0)
            self.weights = weights

    def _convert(self, data):
        """
        Internal function used to check and, if possible, convert the raw data from the user.
        Check that:
            - data is indeed of a list or a np.ndarray;
            - the shape of the data is (NbrPatterns, x), when storing 1D patterns, or (NbrPatterns, x, y),
            when working when 2D patterns;
            - patterns are only made of (-1 and +1) or (0 and 1)
        Convert to:
            - flatten each pattern to 1D array
            - convert to arrays of (-1 and +1)

        :param data: raw data received from the user
        :return: shape - tuple, original shape of the data; data - np.ndarray, converted data
        """
        if type(data) == list:
            data = np.array(data)

        assert type(data) == np.ndarray, 'The type of data should be a list or a np.ndarray'
        assert len(data.shape) in [2, 3], \
            'Data should be a list of patterns you want to store. Its shape should be (NbrPatterns, x(, y))'
        assert np.all(np.logical_or(data == 1, data == -1)) or np.all(np.logical_or(data == 1, data == 0)), \
            'Patterns in data should be vectors of (-1 and +1) or (0 and 1)'

        shape = data[0].shape

        if len(data.shape) == 3:
            nbrPatterns, x, y = data.shape
            data = np.reshape(data, (nbrPatterns, x * y))  # flatten each pattern to 1D array

        data = np.where(data, data, -1)  # convert to arrays of (-1 and +1)

        return shape, data

    def capacity(self):
        """
        Function to print the Network's capacity which is based on the learning_rule used to store the patterns.

        :return: None
        """
        if self.learning_rule == 'Hebbian':
            self._capacity = self.nbr_attractors / (2 * log(self.nbr_attractors))

        elif self.learning_rule == 'Storkey':
            self._capacity = self.nbr_attractors / (sqrt(2 * log(self.nbr_attractors)))

        print('Network\'s capacity is {}'.format(round(self._capacity, 2)))

    def energy(self, state: np.ndarray):
        """
        Function used to return the energy of the hopfield network at a specific state.

        :param state: state of the neurons at a specific timestep
        :return: energy of the hopfield network
        """
        return -0.5 * state.dot(self.weights).dot(state) + np.sum(state * self.threshold)

    def predict(self, data: (list, np.ndarray), nbr_iterations: int=100, sync_update: bool=True,
                activation_fcn: str='sign', threshold: int=0, show: bool=False):
        """
        Function used to submit a new 1D or 2D vector which is often a corrupted version of a stored pattern in order to
        recover this original pattern.

        :param data: list of corrupted patterns that should be recovered
        :param nbr_iterations: number of iterations during with the neurons' state will be updated
        :param sync_update: if True then all neurons are updated at each iteration. If False, a random neuron is
        considered for update at each timestep
        :param activation_fcn: function used for the update of the neurons' state, currently only support 'sign'
        :param threshold: threshold used for the activation function
        :param show: if True then bokeh will be used to show the recovered pattern as well as the energy of the network.
        Only works when one corrupted pattern is submitted
        :return: Ys - list, recovered pattern(s); Es - list, network's energy at the different timestep
        """
        # TODO: Add support for other  'activation_fcn'

        self.threshold = threshold

        _, data = self._convert(data)

        assert data.shape[1] == self.nbr_neurons, 'The dimension of your data is not matching the one of our patterns'

        Xs = data.copy()
        Ys = []
        Es = []

        for x in Xs.copy():
            it = 0
            E = [self.energy(x)]

            while it < nbr_iterations:
                if sync_update:
                    x = np.sign(self.weights.dot(x) - self.threshold)
                else:
                    idx = np.random.randint(0, self.nbr_neurons)
                    x[idx] = np.sign(self.weights[idx].dot(x) - self.threshold)
                E.append(self.energy(x))
                it += 1

            Es.append(E)
            Ys.append(x)

        if show and len(data) == 1:
            self._display(np.array(Xs), np.array(Ys), Es[0])

        return Ys, Es

    def _display(self, X: np.ndarray, Y: np.ndarray, E: list):
        shape = self.original_shape if len(self.original_shape) == 2 else (sqrt(self.original_shape[0]),) * 2

        gridplot = grid(
            [[self._neuron_graph(X, shape, 'Submitted Image'), self._neuron_graph(Y, shape, 'Reconstructed Image')],
             [self._energy_graph(E)]], sizing_mode='stretch_both')
        # gridplot = grid([[gridplot, self._hop_graph()],])
        show(gridplot)

    def _neuron_graph(self, X: np.ndarray, shape: tuple, title: str):
        plot = figure(plot_width=350, plot_height=350)
        plot.title.text = title
        plot.image(image=[np.flipud(X.reshape(shape))], x=0, y=0, dw=shape[0], dh=shape[1], palette=RdBu3)
        return plot

    def _energy_graph(self, E: list):
        plot = figure(plot_width=780, plot_height=350)
        plot.title.text = "Hopfield Network's Energy over time"
        plot.line(list(range(len(E))), E, line_width=2, line_color=RdBu3[2])
        return plot

    def _hop_graph(self):
        plot = figure(plot_width=700, plot_height=700)
        plot.title.text = "Hopfield Network's weights"
        plot.image(image=[self.weights], x=0, y=0, dw=self.weights.shape[0], dh=self.weights.shape[1], palette=RdBu3)
        return plot
