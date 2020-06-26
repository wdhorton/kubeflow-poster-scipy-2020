import pandas as pd


def get_data():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
        sep="\s+",
        na_values="?",
        names=[
            "mpg",
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model year",
            "origin",
            "car name",
        ],
    )
    return df


if __name__ == "__main__":
    df = get_data()
    df.to_pickle("raw_df.pkl")
