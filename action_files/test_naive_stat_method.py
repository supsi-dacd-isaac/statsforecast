import pytest
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, StatNaive
from statsforecast.utils import ConformalIntervals
import matplotlib.pyplot as plt
import numpy as np


@pytest.mark.parametrize("stat", ['mean', 'median'])
def test_unit_prediction_step(stat):
    df = pd.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')
    df = df.iloc[:748, :]
    df.plot(); plt.show()
    h = 24
    intervals = ConformalIntervals(h=h, n_windows=df.shape[0], method='naive_error')
    sf = StatsForecast(models=[StatNaive(prediction_intervals=intervals, step=1, stat=stat)], freq=1, n_jobs=1)
    sf.fit(df=df)
    preds = sf.predict(h=h, level=np.linspace(10, 95, 4))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    preds.iloc[:, 2:].plot(ax=ax)
    ax.plot(np.ones(len(preds))*df['y'].aggregate(stat), label=stat, linestyle='--')

    plt.show()