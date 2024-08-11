!pip install darts
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.ad import ForecastingAnomalyModel, NormScorer
from darts.models import TCNModel

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from darts import TimeSeries
from darts.ad.utils import (
    eval_metric_from_binary_prediction,
    eval_metric_from_scores,
    show_anomalies_from_scores,
)
from darts.ad import (
    ForecastingAnomalyModel,
    KMeansScorer,
    NormScorer,
    WassersteinScorer,
)
from darts.dataprocessing.transformers import Scaler
from darts.datasets import TaxiNewYorkDataset
from darts.metrics import mae, rmse
from darts.models import RegressionModel

# load the data
series_taxi = TaxiNewYorkDataset().load()

# define start and end dates for some known anomalies
anomalies_day = {
    "NYC Marathon": ("2014-11-02 00:00", "2014-11-02 23:30"),
    "Thanksgiving ": ("2014-11-27 00:00", "2014-11-27 23:30"),
    "Christmas": ("2014-12-24 00:00", "2014-12-25 23:30"),
    "New Years": ("2014-12-31 00:00", "2015-01-01 23:30"),
    "Snow Blizzard": ("2015-01-26 00:00", "2015-01-27 23:30"),
}
anomalies_day = {
    k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in anomalies_day.items()
}

# create a series with the binary anomaly flags
anomalies = pd.Series([0] * len(series_taxi), index=series_taxi.time_index)
for start, end in anomalies_day.values():
    anomalies.loc[(start <= anomalies.index) & (anomalies.index <= end)] = 1.0

series_taxi_anomalies = TimeSeries.from_series(anomalies)

# plot the data and the anomalies
fig, ax = plt.subplots(figsize=(15, 5))
series_taxi.plot(label="Number of taxi passengers", linewidth=1, color="#6464ff")
(series_taxi_anomalies * 10000).plot(label="5 known anomalies", color="r", linewidth=1)
plt.show()

def plot_anom(selected_anomaly, delta_plotted_days):
    one_day = series_taxi.freq * 24 * 2
    anomaly_date = anomalies_day[selected_anomaly][0]
    start_timestamp = anomaly_date - delta_plotted_days * one_day
    end_timestamp = anomaly_date + (delta_plotted_days + 1) * one_day

    series_taxi[start_timestamp:end_timestamp].plot(
        label="Number of taxi passengers", color="#6464ff", linewidth=0.8
    )

    (series_taxi_anomalies[start_timestamp:end_timestamp] * 10000).plot(
        label="Known anomaly", color="r", linewidth=0.8
    )
    plt.title(selected_anomaly)
    plt.show()


for anom_name in anomalies_day:
    plot_anom(anom_name, 3)
    break  # remove this to see all anomalies
    
from darts.models import TCNModel

# Split the data into training and testing sets
s_taxi_train = series_taxi[:4500]
s_taxi_test = series_taxi[4500:]

# Add covariates (hour and day of the week)
add_encoders = {
    "cyclic": {"future": ["hour", "dayofweek"]},
}

# One week corresponds to (7 days * 24 hours * 2) of 30 minutes
one_week = 7 * 24 * 2

# Create a TCN model
tcn_model = TCNModel(
    input_chunk_length=one_week,
    output_chunk_length=1,
    kernel_size=3,
    num_filters=32,
    dropout=0.1,
    add_encoders=add_encoders
)

# Fit the TCN model using the training data
tcn_model.fit(s_taxi_train)

# Optionally, you can forecast and evaluate the model on the test data
# forecast = tcn_model.predict(n=len(s_taxi_test), series=s_taxi_train)
# print(forecast)
