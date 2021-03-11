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
    # axis=1 remove columns
    # inplace makes the variable data be overwritten
    data.drop(['id', 'date', 'query', 'user'], axis=1, inplace=True)

    def clean_tweet(tweet):
        tweet = BeautifulSoup(tweet).get_text()
        # Remove the @ and its mention
        tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
        # Remove the links of URLs
        tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
        # Nos quedamos solamente con los caracteres
        tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
        # Remove extra blanks
        tweet = re.sub(r" +", ' ', tweet)
        return tweet