import json

from joblib import load
import pandas as pd
from sklearn.metrics import mean_squared_error


def evaluate(regr, df, target_column):
    X_test = df.drop(target_column, axis="columns")
    y_test = df[target_column]
    y_pred = regr.predict(X_test)
    return mean_squared_error(y_test, y_pred)


if __name__ == "__main__":
    regr = load("trained_model.joblib")
    val_df = pd.read_pickle("val_df.pkl")

    mse = evaluate(regr, val_df, "mpg")
    print(f"MSE: {mse}")

    metrics = {"metrics": [{"name": "mse", "numberValue": mse, "format": "RAW",},]}
    print(metrics)

    with open("/mlpipeline-metrics.json", "w") as f:
        json.dump(metrics, f)
