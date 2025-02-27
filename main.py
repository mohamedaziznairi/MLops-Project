import argparse
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
# Set MLflow tracking URI to use SQLite
#mlflow.set_tracking_uri("sqlite:///mlflow.db")
app = FastAPI()  # This is the FastAPI app instance
def main():
   # Test comment 
    # Initialize MLflow experiment
    mlflow.set_experiment("MLflow_Atelier5")
   

    # Create an argument parser for CLI
    parser = argparse.ArgumentParser()

    # Add CLI arguments
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the dataset")

    # Parse arguments
    args = parser.parse_args()

    # Prepare data
    if args.prepare:
        x_train, x_test, y_train, y_test = prepare_data(args.file_path)
        print("✅ Data prepared.")

    # Train model with MLflow tracking
    if args.train:
        x_train, x_test, y_train, y_test = prepare_data(args.file_path)
        with mlflow.start_run():
            model = train_model(x_train, y_train)

            # Log parameters (example: hyperparameters)
            mlflow.log_param("train_size", len(x_train))
            mlflow.log_param("test_size", len(x_test))

            # Save model using MLflow
            mlflow.sklearn.log_model(model, "model")
            save_model(model)

            print("✅ Model trained and logged in MLflow.")

    # Evaluate model with MLflow tracking
    if args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data(args.file_path)
        model = load_model()

        with mlflow.start_run():
            metrics = evaluate_model(model, x_test, y_test)


        # Log numerical metrics in MLflow
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])

        # Log the classification report as an artifact (since it's a dictionary)
        class_report_path = "classification_report.json"
        with open(class_report_path, "w") as f:
            json.dump(metrics["class_report"], f)

        mlflow.log_artifact(class_report_path)
        print("✅ Model evaluated and metrics logged in MLflow.")

# Run the script
if __name__ == "__main__":
    main()