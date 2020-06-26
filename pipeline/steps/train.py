import argparse
from distutils.util import strtobool

import pandas as pd
from sklearn import linear_model
from joblib import dump


def train(df, target_column, normalize):
    X = df.drop(target_column, axis="columns")
    y = df[target_column]
    regr = linear_model.LinearRegression(normalize=normalize)

    print("Training model")
    regr.fit(X, y)
    return regr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--normalize", help="normalize arg for LinearRegression")

    args = parser.parse_args()

    train_df = pd.read_pickle("train_df.pkl")
    regr = train(train_df, "mpg", strtobool(args.normalize))
    dump(regr, "trained_model.joblib")
