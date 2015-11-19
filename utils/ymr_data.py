import pandas as pd
import numpy as np
import random
import sklearn.cross_validation
from collections import Counter
import itertools
import tarfile

def load(filename='../data/yahoo-movie-reviews.json'):
    with tarfile.open('../data/yahoo-movie-reviews.json.tar.gz', 'r:gz') as tf:
        f = tf.extractfile(tf.getmembers()[0])
        data = pd.read_json(f)
        data.movieName = data.movieName.str.strip()
        data.text = data.text.str.strip()
        data.title = data.title.str.strip()
        data = data[data.text.str.len() > 0]
        data.url = data.url.str.strip()
        return data

def make_polar(data, balance=True):
    data_polar = data.loc[data.rating != 3].copy()
    data_polar.loc[data_polar.rating <= 2, 'rating'] = 0
    data_polar.loc[data_polar.rating >= 4, 'rating'] = 1
    if balance:
        # Subsample - We want the same number of positive and negative examples
        K = data_polar.groupby('rating').rating.count().min()
        indices = itertools.chain(*[np.random.choice(v, K, replace=False) for k, v in  data_polar.groupby('rating').groups.items()])
        data_polar = data_polar.reindex(indices).copy()
    return data_polar

def vocab(df):
    character_counts = Counter(iter(df.text.str.cat()))
    vocabulary = { x[0]: i for i, x in enumerate(character_counts.most_common()) }
    vocabulary_inv = [x[0] for x in character_counts.most_common()]
    return [vocabulary, vocabulary_inv]

def train_test_split(df, train_size=0.8, random_state=0):
    train, test = sklearn.cross_validation.train_test_split(df,
        train_size=train_size, stratify=df.rating, random_state=random_state)
    return [train, test] 

def make_xy(df, vocabulary):
    x = np.array([[vocabulary[x] for x in review_text] for review_text in df.text])
    y = np.array([y for y in df.rating])
    return [x, y]                