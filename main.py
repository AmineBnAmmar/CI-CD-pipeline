import mlflow
import time 
import mlflow.system_metrics
import numpy as np
import psutil  # Import psutil for system metrics
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def get_system_metrics():
    """Returns system resource usage metrics."""
    return {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_used_percent": psutil.virtual_memory().percent,
        "disk_used_percent": psutil.disk_usage("/").percent,
    }


def main():
    # Configure MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Churn Prediction")
    # Enable automatic logging of system metrics at a specified interval (in seconds)
    mlflow.enable_system_metrics_logging()
    # Start MLflow run
    with mlflow.start_run(run_name="XGBoost Baseline"):

        time.sleep(15)
        # Log system metrics before training
        print("\n=== Logging system metrics before training ===")
        mlflow.log_metrics(get_system_metrics())

        # Data preparation
        print("\n=== Preparing data ===")
        X_train, y_train, X_test, y_test, encoder, scaler = prepare_data()

        # Log data parameters
        mlflow.log_params(
            {
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "num_features": X_train.shape[1],
                "class_ratio": f"{sum(y_train)/len(y_train):.2f}",
            }
        )

        # Model training
        print("\n=== Training XGBoost model ===")
        model_params = {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5}
        model = train_model(X_train, y_train, **model_params)

        # Log model parameters
        mlflow.log_params(model_params)

        # Log system metrics after training
        print("\n=== Logging system metrics after training ===")
        mlflow.log_metrics(get_system_metrics())

        # Model evaluation
        print("\n=== Evaluating model ===")
        accuracy, classification_report = evaluate_model(model, X_test, y_test)

        # Log evaluation metrics
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": classification_report["weighted avg"]["precision"],
                "recall": classification_report["weighted avg"]["recall"],
                "f1_score": classification_report["weighted avg"]["f1-score"],
            }
        )

        # Log system metrics after evaluation
        print("\n=== Logging system metrics after evaluation ===")
        mlflow.log_metrics(get_system_metrics())

        # Save and log model artifacts
        print("\n=== Saving model ===")
        save_model(model, encoder, scaler, "churn_model.pkl")
        mlflow.log_artifact("churn_model.pkl")

        # Log model to MLflow registry
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="XGBoost_Churn_Predictor",
        )

        # Model loading demonstration
        print("\n=== Loading saved model ===")
        loaded_model, loaded_encoder, loaded_scaler = load_model("churn_model.pkl")

        # Example prediction
        print("\n=== Sample prediction ===")
        sample_idx = 0
        sample_data = X_test[sample_idx].reshape(1, -1)
        sample_pred = loaded_model.predict(sample_data)
        print(f"Predicted: {sample_pred[0]} | Actual: {y_test.iloc[sample_idx]}")

        # Log prediction example
        mlflow.log_dict(
            {
                "sample_prediction": {
                    "features": X_test[sample_idx].tolist(),
                    "prediction": int(sample_pred[0]),
                    "actual": int(y_test.iloc[sample_idx]),
                }
            },
            "sample_prediction.json",
        )

        # Log final system metrics
        print("\n=== Logging final system metrics ===")
        mlflow.log_metrics(get_system_metrics())

        print("\n=== MLflow Run ID ===")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("\n=== Workflow completed! ===")


if __name__ == "__main__":
    main()
# Testing GitHub Actions
# Testing GitHub Actions
