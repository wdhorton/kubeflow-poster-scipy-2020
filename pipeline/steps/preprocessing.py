import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(df):
    # fill unknown values in column
    df = df.fillna({"horsepower": df["horsepower"].median()})

    # drop unused columns
    df = df.drop(["origin", "car name"], axis="columns")

    # split 20% for test set
    return train_test_split(df, test_size=0.2, random_state=42)


if __name__ == "__main__":
    raw_df = pd.read_pickle("raw_df.pkl")

    train_df, val_df = preprocess(raw_df)

    train_df.to_pickle("train_df.pkl")
    val_df.to_pickle("val_df.pkl")
