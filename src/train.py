import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

NAME = "Vaishnav Nigade"
ROLL_NO = "2022BCD0045"


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    params = load_params()

    dataset_path = params.get("dataset_path", "data/winequality.csv")
    df = pd.read_csv(dataset_path, sep=";")

    df["target"] = (df["quality"] >= 6).astype(int)

    selected_features = params["features"]
    X = df[selected_features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y
    )

    model_type = params["model_type"]

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"]
        )
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=params["max_depth"],
            random_state=params["random_state"]
        )
    else:
        raise ValueError("Unsupported model type")

    mlflow.set_experiment("2022BCD0045_experiment")

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("n_estimators", params.get("n_estimators", None))
        mlflow.log_param("test_size", params["test_size"])
        mlflow.log_param("random_state", params["random_state"])
        mlflow.log_param("features", ",".join(selected_features))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        model_path = "models/model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        metrics = {
            "name": NAME,
            "roll_no": ROLL_NO,
            "dataset_path": dataset_path,
            "model_type": model_type,
            "features": selected_features,
            "accuracy": accuracy,
            "f1_score": f1
        }

        with open("metrics/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact("metrics/metrics.json")

        print("Training completed")
        print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()