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
    #
    # init.autocorrelation has the following format: number_of_series,start,end
    # which means that we will be plotting an autocorrelation for number_of_series random series from start to end
    #
    # Here we are transforming this series into a list of integers if possible
    #
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


















































































































































