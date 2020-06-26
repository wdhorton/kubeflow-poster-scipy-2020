import pandas as pd


PREVIOUS_WEIGHT_MEAN = 2960
TOLERANCE = 0.05


def data_validation(df):
    change_in_mean = abs(PREVIOUS_WEIGHT_MEAN - df["weight"].mean())

    if change_in_mean / PREVIOUS_WEIGHT_MEAN > TOLERANCE:
        raise Exception("error detected in data validation")


if __name__ == "__main__":
    raw_df = pd.read_pickle("raw_df.pkl")
    data_validation(raw_df)
