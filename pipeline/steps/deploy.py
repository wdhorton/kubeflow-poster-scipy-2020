import argparse
import json

from joblib import dump, load

PREVIOUS_MSE = 12


def deploy(regr, mse):
    if mse < PREVIOUS_MSE:
        print("Saving model")
        dump(regr, "deployed_model.joblib")

        # In reality you'd probably also have something like:
        # import boto3
        # s3 = boto3.resource("s3")
        # s3.meta.client.upload_file("deployed_model.joblib", "mybucket", "deployed_model.joblib")
    else:
        print("No improvement, skipping deploy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--metrics", help="JSON containing metrics")

    args = parser.parse_args()

    metrics = json.loads(args.metrics)
    mse = metrics["metrics"][0]["numberValue"]

    regr = load("trained_model.joblib")

    deploy(regr, mse)
