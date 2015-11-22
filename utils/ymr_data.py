import pandas as pd
import numpy as np
import random
from sklearn.cross_validation import train_test_split
from collections import Counter
import itertools
import tarfile


def load_ymr_data(filename='../data/yahoo-movie-reviews.json'):
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
        grouped_ratings = data_polar.groupby('rating')
        K = grouped_ratings.rating.count().min()
        indices = itertools.chain(
            *[np.random.choice(v, K, replace=False) for k, v in grouped_ratings.groups.items()])
        data_polar = data_polar.reindex(indices).copy()
    return data_polar


def make_vocab(df):
    character_counts = Counter(iter(df.text.str.cat()))
    vocabulary = {x[0]: i for i, x in enumerate(character_counts.most_common())}
    vocabulary_inv = [x[0] for x in character_counts.most_common()]
    return [vocabulary, vocabulary_inv]


def make_xy(df, vocabulary, one_hot=True):
    x = np.array([[vocabulary[x] for x in review_text] for review_text in df.text])
    y = np.array([y for y in df.rating])
    if one_hot:
        y_onehot = np.zeros((len(y), 2))
        y_onehot[np.arange(len(y)), y] = 1.
        y = y_onehot
    return [x, y]


def generate_dataset(fixed_length=None, one_hot=True, test_size=0.2, dev_size=0.05, random_state=10):
    # Load data
    df = load_ymr_data()

    # Optionally pad all sentences
    if fixed_length:
        padding_character = u"\u0000"
        df.text = df.text.str.slice(0, fixed_length)
        df.text = df.text.str.ljust(fixed_length, padding_character)

    # Generate vocabulary and dataset
    vocab, vocab_inv = make_vocab(df)
    data = make_polar(df)

    #  Split data into train, dev, and text
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    train, dev = train_test_split(train, test_size=dev_size, random_state=random_state)

    # Generate inputs and labels
    train_x, train_y = make_xy(train, vocab, one_hot=one_hot)
    dev_x, dev_y = make_xy(dev, vocab, one_hot=one_hot)
    test_x, test_y = make_xy(test, vocab, one_hot=one_hot)

    # Return
    return [train_x, train_y, dev_x, dev_y, test_x, test_y]
