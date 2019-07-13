# Plotting Script

import numpy as np
import matplotlib.pyplot as plt
import gin
import logging
from pandas.plotting import autocorrelation_plot
import matplotlib
matplotlib.rcParams.update({"font.size": 14})

log = logging.getLogger("Chronos")


@gin.configurable
def auto_correlation_plot(data, number_of_series, start, end, save_plot=None):
    # Plot auto correlations in data
    s = [number_of_series, start, end]

    fig = plt.figure()

    log.debug("Plotting autocorrelation for %d random timeseries out of %d. Timeslot from %d to %d",
              number_of_series, number_of_series, start, end)

    series = np.random.choice(range(number_of_series), number_of_series, replace=False)
    for i in series:
        autocorrelation_plot(data[start:end, i])

    fig.canvas.set_window_title('Auto Correlation')
    plt.show()

    if save_plot is not None:
        log.debug("Saving autocorrelation plot to: %s", save_plot + "_autocorrelation.png")
        fig.savefig(save_plot + "_autocorrelation.png")


@gin.configurable
def plot_prediction(
        data,
        start_plot,
        end_plot,
        train_predict,
        validation_predict,
        test_predict,
        series,
        save_plot=None,
        plot_show=True
):
    """
    Plot the predictions and the true data.
    Args:
        data(AbstractInputGenerator):
        start_plot(int): Start of the sequence to plot
        end_plot(int): End of the sequence to plot
        train_predict(np.array): Shape [n_train, num_time_series]
        validation_predict(np.array): Shape [n_val, num_time_series]
        test_predict(np.array): Shape [n_test, num_time_series]
        series(int): The series to plot
        save_plot(str): Path for image
        plot_show(bool): Show the plot

    Returns:

    """
    assert series < data.num_time_series - 1, "Data has only {} time series. Requested series {}".format(
        data.num_time_series, series
    )
    assert len(train_predict.shape) == 2, "Expected Rank two for train_predict but got {}".format(
        len(train_predict.shape)
    )
    assert len(validation_predict.shape) == 2, "Expected Rank two for validation_predict but got {}".format(
        len(validation_predict.shape)
    )
    assert len(test_predict.shape) == 2, "Expected Rank two for test_predict but got {}".format(
        len(test_predict.shape)
    )
    n = len(data)
    #
    # Create empty series of the same length of the data and set the values to nan
    # This way, we can fill the appropriate section for train, valid, test so that
    # when we print them, they appear at the appropriate loction with respect to the original timeseries
    #
    train_predict_plot = np.empty((n, 1))
    train_predict_plot[:, :] = np.nan
    validation_predict_plot = np.empty((n, 1))
    validation_predict_plot[:, :] = np.nan
    test_prediction_plot = np.empty((n, 1))
    test_prediction_plot[:, :] = np.nan

    #
    # We use window data to predict a value at horizon from the end of the window, therefore start is
    # is at the end of the horizon
    #
    if train_predict is not None:
        train_predict_plot[data.train_range, 0] = train_predict[:, series]

    if validation_predict is not None:
        validation_predict_plot[data.validation_range, 0] = validation_predict[:, series]

    if test_predict is not None:
        test_prediction_plot[data.test_range, 0] = test_predict[:, series]

    # Plotting the original series and whatever is available of trainPredictPlot, validPredictPlot and testPredictPlot
    fig = plt.figure()

    plt.plot(data[start_plot:end_plot, series], label="True Data", alpha=0.8)
    plt.plot(train_predict_plot[start_plot:end_plot], label="Train Set Prediction")
    plt.plot(validation_predict_plot[start_plot:end_plot], label="Val Set Prediction")
    plt.plot(test_prediction_plot[start_plot:end_plot], label="Test Set Prediction")
    plt.legend()
    plt.ylabel("Timeseries")
    plt.xlabel("Time")
    plt.title("Prediction Plotting for timeseries # %d" % (series))

    fig.canvas.set_window_title('Prediction')
    if plot_show:
        plt.show()

    if save_plot is not None:
        log.debug("Saving prediction plot to: %s", save_plot + "_prediction.png")
        fig.savefig(save_plot + "_prediction.png")
