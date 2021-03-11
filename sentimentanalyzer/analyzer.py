import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

def init():
    #columns for the dataframe, the csv file does not have them explicitly
    cols = ["sentiment", "id", "date", "query", "user", "text"]

    train_data = pd.read_csv(
        "sentimentanalyzer/data/test.csv",
        header=None,
        names=cols,
        engine="python",
        encoding="latin1"
    )

    test_data = pd.read_csv(
        "sentimentanalyzer/data/test.csv",
        header=None,
        names=cols,
        engine="python",
        encoding="latin1"
    )
    data = train_data
    # remove the headers that we are not going to use
    data.drop(['id', 'date', 'query', 'user'], axis=1, inplace=True)