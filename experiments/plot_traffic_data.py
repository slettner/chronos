# Check out the traffic data
# Data set from the paper

import numpy as np
import chronos.plots as plt
from chronos.data import from_txt
from chronos.data import NumpyInputGenerator
DATA_PATH = "../data/traffic-subset.txt"


def fake_prediction_plot(data):

    data = NumpyInputGenerator(data=data, train=0.8, validation=0.1, batch_size=1, window=24, horizon=1)
    data_prediction = data.data + np.random.normal(loc=0, scale=0.03, size=(len(data), 862))
    train_predict = data_prediction[data.train_range]
    val_predict = data_prediction[data.validation_range]
    test_predict = data_prediction[data.test_range]
    plt.plot_prediction(data, start_plot=0, end_plot=200,
                        train_predict=train_predict,
                        validation_predict=val_predict,
                        test_predict=test_predict,
                        series=1
    )


if __name__ == '__main__':
    data = from_txt(path=DATA_PATH)
    plt.auto_correlation_plot(data, number_of_series=20, start=0, end=200 + 7*24)
    fake_prediction_plot(data)
