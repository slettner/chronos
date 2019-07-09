# Check out the traffic data
# Data set from the paper

import chronos.plots as plt
from chronos.data import from_txt

DATA_PATH = "../data/traffic.txt"


if __name__ == '__main__':
    data = from_txt(path=DATA_PATH)
    plt.auto_correlation_plot(data, number_of_series=20, start=1000, end=1350)
