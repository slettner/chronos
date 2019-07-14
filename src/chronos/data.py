# Implements

import gin
import abc
import numpy as np
import tensorflow as tf
import functools
import multiprocessing as mp


class MODE:
    TRAIN = 0
    EVAL = 1
    TEST = 2


@gin.configurable
def from_txt(path, delimiter=','):
    """
    Load numpy array from txt file

    Args:
        path(str): Path to the .txt file
        delimiter(char): The delimiter.

    Returns:
        np.array
    """
    return np.loadtxt(path, delimiter=delimiter)


class AbstractInputGenerator(metaclass=abc.ABCMeta):
    """ Interface for input generators for the tensorflow estimator api. """

    def __init__(self, train, validation, batch_size, window, horizon, normalize):
        """
        Abstract Class

        Args:
            window(int): Size of the window for inputs.
            horizon(int): Distance in time between the last input and the prediction
            train(float): Between 0 and 1. Percentage of the training set.
            validation(float): Between 0 and 1.Percentage of the validation set.
            batch_size(int): The batch size for training

        """
        assert 0 < train <= 1, "Train must be in (0, 1]"
        assert 0 < validation <= 1, "Validation must be in (0, 1]"
        assert train + validation <= 1, "The sum of train and validation must be smaller or equal to one"

        self.train = train
        self.validation = validation
        self.window = window
        self.horizon = horizon
        self.normalize = normalize
        self.batch_size = batch_size

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, slice):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_time_series(self):
        """ The number of time series variable """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def train_range(self):
        """ The amount of training samples """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def validation_range(self):
        """ The amount of eval samples """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def test_range(self):
        """ The amount of test samples """
        raise NotImplementedError

    @abc.abstractmethod
    def _input_fn(self, mode):
        """
        Implements the input function according to the tensorflow estimator api

        Args:
            mode(MODE): Return training, validation or test data.

        Returns:
            dataset
        """
        raise NotImplementedError

    def __call__(self, mode, epochs=None):
        """

        Args:
            mode(MODE): Return training, validation or test data.

        """
        # return the input function with the correct mode.
        # Return value is a function with signature
        return functools.partial(self._input_fn, mode=mode, epochs=epochs)


@gin.configurable
class NumpyInputGenerator(AbstractInputGenerator):

    """ Wrapper class for tf.estimator numpy_input_fn.
        Note: requires that all data fits into memory twice!
    """

    def __init__(self, data, train, validation, batch_size,
                 window, horizon, normalize=True):
        """

        Args:
            data(np.array): The data. Shape: time, feature
            window(int): Size of the window for inputs.
            horizon(int): Distance in time between the last input and the prediction
            train(float): Between 0 and 1. Percentage of the training set.
            validation(float): Between 0 and 1.Percentage of the validation set.
            normalize(bool): Noramlize. (per feature)
            batch_size(int): The batch size for training.
        """
        super(NumpyInputGenerator, self).__init__(
            window=window,
            horizon=horizon,
            normalize=normalize,
            train=train,
            validation=validation,
            batch_size=batch_size
        )
        assert len(data.shape) == 2, "The data should have shape time x feature"
        self.data = data.astype(np.float32)
        self.num_samples, self.feature_dim = self.data.shape
        tf.logging.info("Received data with {} values and {} time series values".format(
            self.num_samples, self.feature_dim
        ))
        self.scale = np.ones(self.feature_dim)
        if normalize:
            self._normalize()

        # each is a tuple of X, Y
        self.train_data, self.validation_data, self.test_data = self._split_data()

    def _normalize(self):
        """ Normalize the raw data """
        num_train_sample = int(self.num_samples * self.train)

        for i in range(self.feature_dim):
            # only look at the training data for normalization
            self.scale[i] = np.max(np.abs(self.data[0:num_train_sample, i]))
            # now normalize all data
            self.data[:, i] = self.data[:, i] / self.scale[i]

    def _split_data(self):
        """ Split the data set into train, val and test """

        train = self.train
        validation = self.validation

        train_set = range(self.window + self.horizon - 1, int(train * self.num_samples))
        valid_set = range(int(train * self.num_samples), int((train + validation) * self.num_samples))
        test_set = range(int((train + validation) * self.num_samples), self.num_samples)

        tf.logging.info("Number of training samples: {}".format(len(train_set)))
        tf.logging.info("Number of validation samples: {}".format(len(valid_set)))
        tf.logging.info("Number of test samples: {}".format(len(test_set)))

        training_data = self._get_data(train_set)
        validation_data = self._get_data(valid_set)
        test_data = self._get_data(test_set)

        return training_data, validation_data, test_data

    def _get_data(self, idx_range):
        """
        Take the given range out of the data set and reshape it.

        Returns:
            X(np.array), Y(np.array)
        """
        n = len(idx_range)

        X = np.zeros((n, self.window, self.feature_dim))
        Y = np.zeros((n, self.feature_dim))

        # assigning the inputs the correct labels
        for i in range(n):
            end = idx_range[i] - self.horizon + 1
            start = end - self.window

            X[i, :, :] = self.data[start:end, :]
            Y[i, :] = self.data[idx_range[i], :]

        return X.astype(np.float32), Y.astype(np.float32)

    def _input_fn(self, mode, epochs=None):
        """ Implement the input function for the tf.estimator api """

        if mode == MODE.TRAIN:
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x=self.train_data[0],
                y=self.train_data[1],
                batch_size=self.batch_size if epochs is None else 1,
                num_epochs=epochs,
                shuffle=False,
                num_threads=1
            )
            return input_fn()

        elif mode == MODE.EVAL:
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x=self.validation_data[0],
                y=self.validation_data[1],
                batch_size=len(self.validation_range) if epochs is None else 1,
                num_epochs=epochs,
                shuffle=False,
                num_threads=1
            )
            return input_fn()

        elif mode == MODE.TEST:
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x=self.test_data[0],
                y=self.test_data[1],
                batch_size=len(self.test_range) if epochs is None else 1,
                num_epochs=epochs,
                shuffle=False,
                num_threads=1
            )
            return input_fn()

    @property
    def train_range(self):
        return range(self.window + self.horizon - 1, int(self.train * self.num_samples))

    @property
    def validation_range(self):
        return range(int(self.train * self.num_samples), int((self.train + self.validation) * self.num_samples))

    @property
    def test_range(self):
        return range(int((self.train + self.validation) * self.num_samples), self.num_samples)

    @property
    def num_time_series(self):
        return self.feature_dim

    def __getitem__(self, slice):
        return self.data[slice]

    def __len__(self):
        return self.num_samples
