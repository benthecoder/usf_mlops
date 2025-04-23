# src/trainingflowgcp.py
from metaflow import (
    FlowSpec,
    step,
    Parameter,
    kubernetes,
    conda_base,
    retry,
    timeout,
    catch,
)
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@conda_base(
    python="3.11",
    libraries={
        "pandas": "2.2.2",
        "numpy": "1.26.4",
        "scikit-learn": "1.5.1",
        "mlflow": "2.15.1",
        "gcsfs": "2025.3.2",  # For reading from Google Cloud Storage
        "google-cloud-storage": "3.1.0",  # For directly uploading model to GCS
    },
)
class CardioTrainingFlow(FlowSpec):
    """
    A flow to train models for cardiovascular disease prediction.
    """

    # Parameters for the flow
    data_path = Parameter(
        "data_path",
        default="gs://mlops-457620-cardio-data/cardio/",
        help="Path to the cardio data directory in GCS",
    )

    random_seed = Parameter(
        "random_seed", default=42, help="Random seed for reproducibility"
    )

    n_estimators = Parameter(
        "n_estimators", default=100, help="Number of estimators for Random Forest"
    )

    use_selected_features = Parameter(
        "use_selected_features",
        default=True,
        help="Whether to use feature-selected datasets",
    )

    @kubernetes
    @retry(times=3)
    @timeout(minutes=10)
    @step
    def start(self):
        """
        Load the training, validation, and test data.
        """
        import pandas as pd

        # Choose dataset based on parameter
        dataset_suffix = "_selected" if self.use_selected_features else ""

        # Load data
        self.train_data = pd.read_csv(f"{self.data_path}train_data{dataset_suffix}.csv")
        self.val_data = pd.read_csv(f"{self.data_path}val_data{dataset_suffix}.csv")
        self.test_data = pd.read_csv(f"{self.data_path}test_data{dataset_suffix}.csv")

        # Separate features and target
        self.X_train = self.train_data.drop("cardio", axis=1)
        self.y_train = self.train_data["cardio"]

        self.X_val = self.val_data.drop("cardio", axis=1)
        self.y_val = self.val_data["cardio"]

        self.X_test = self.test_data.drop("cardio", axis=1)
        self.y_test = self.test_data["cardio"]

        print(f"Training data shape: {self.X_train.shape}")
        print(f"Validation data shape: {self.X_val.shape}")
        print(f"Test data shape: {self.X_test.shape}")

        # Go to next steps in parallel
        self.next(self.train_logistic, self.train_rf)

    @kubernetes
    @retry(times=3)
    @timeout(minutes=30)
    @catch(var="train_error")
    @step
    def train_logistic(self):
        """
        Train a logistic regression model.
        """
        from sklearn.linear_model import LogisticRegression

        # Train logistic regression model
        self.model = LogisticRegression(random_state=self.random_seed, max_iter=1000)
        self.model.fit(self.X_train, self.y_train)

        # Evaluate on validation set
        y_pred = self.model.predict(self.X_val)
        self.val_accuracy = accuracy_score(self.y_val, y_pred)
        self.val_f1 = f1_score(self.y_val, y_pred)
        self.val_auc = roc_auc_score(
            self.y_val, self.model.predict_proba(self.X_val)[:, 1]
        )

        print(f"Logistic Regression Validation Metrics:")
        print(f"Accuracy: {self.val_accuracy:.4f}")
        print(f"F1 Score: {self.val_f1:.4f}")
        print(f"ROC AUC: {self.val_auc:.4f}")

        self.model_type = "logistic_regression"
        self.next(self.choose_model)

    @kubernetes
    @retry(times=3)
    @timeout(minutes=60)
    @catch(var="train_error")
    @step
    def train_rf(self):
        """
        Train a Random Forest classifier.
        """
        from sklearn.ensemble import RandomForestClassifier

        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_seed,
            n_jobs=-1,  # Use all available cores
        )
        self.model.fit(self.X_train, self.y_train)

        # Evaluate on validation set
        y_pred = self.model.predict(self.X_val)
        self.val_accuracy = accuracy_score(self.y_val, y_pred)
        self.val_f1 = f1_score(self.y_val, y_pred)
        self.val_auc = roc_auc_score(
            self.y_val, self.model.predict_proba(self.X_val)[:, 1]
        )

        print(f"Random Forest Validation Metrics:")
        print(f"Accuracy: {self.val_accuracy:.4f}")
        print(f"F1 Score: {self.val_f1:.4f}")
        print(f"ROC AUC: {self.val_auc:.4f}")

        self.model_type = "random_forest"
        self.next(self.choose_model)

    @kubernetes
    @retry(times=2)
    @timeout(minutes=10)
    @step
    def choose_model(self, inputs):
        """
        Choose the best model based on validation AUC.
        """
        # Compare models based on validation AUC
        models = []
        for inp in inputs:
            if hasattr(inp, "train_error") and inp.train_error:
                print(
                    f"Model {inp.model_type if hasattr(inp, 'model_type') else 'unknown'} failed: {inp.train_error}"
                )
                continue
            models.append(
                {
                    "model": inp.model,
                    "type": inp.model_type,
                    "auc": inp.val_auc,
                    "accuracy": inp.val_accuracy,
                    "f1": inp.val_f1,
                }
            )

        if not models:
            raise ValueError("All models failed to train!")

        # Select the model with the best validation AUC
        self.best_model = max(models, key=lambda x: x["auc"])

        # Explicitly set the conflicting artifacts
        self.model = self.best_model["model"]
        self.val_auc = self.best_model["auc"]
        self.val_accuracy = self.best_model["accuracy"]
        self.val_f1 = self.best_model["f1"]
        self.model_type = self.best_model["type"]

        # Now merge the remaining artifacts
        self.merge_artifacts(inputs)

        print(f"Selected model: {self.best_model['type']}")
        print(f"Validation AUC: {self.best_model['auc']:.4f}")

        self.next(self.evaluate_model)

    @kubernetes
    @retry(times=2)
    @timeout(minutes=15)
    @step
    def evaluate_model(self):
        """
        Evaluate the best model on the test set.
        """
        # Evaluate on test set
        y_pred = self.model.predict(self.X_test)
        self.test_accuracy = accuracy_score(self.y_test, y_pred)
        self.test_f1 = f1_score(self.y_test, y_pred)
        self.test_auc = roc_auc_score(
            self.y_test, self.model.predict_proba(self.X_test)[:, 1]
        )

        print(f"Test Metrics:")
        print(f"Accuracy: {self.test_accuracy:.4f}")
        print(f"F1 Score: {self.test_f1:.4f}")
        print(f"ROC AUC: {self.test_auc:.4f}")

        self.next(self.register_model)

    @kubernetes
    @retry(times=3)
    @timeout(minutes=20)
    @step
    def register_model(self):
        """
        Register the best model with MLflow.
        """
        import mlflow
        import os

        # Try setting username/password for MLflow authentication
        os.environ["MLFLOW_TRACKING_USERNAME"] = "mlflow"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "mlflow"

        # Set up MLflow tracking
        mlflow_uri = "https://mlflow-server-824469552969.us-west2.run.app"
        print(f"Connecting to MLflow at: {mlflow_uri}")
        mlflow.set_tracking_uri(mlflow_uri)

        try:
            # Print MLflow version for debugging
            print(f"MLflow version: {mlflow.__version__}")

            mlflow.set_experiment("cardio-disease-metaflow")

            # Start MLflow run and log everything
            with mlflow.start_run(run_name=f"metaflow-gcp-{self.model_type}"):
                # Log parameters
                mlflow.log_param("model_type", self.model_type)
                mlflow.log_param("use_selected_features", self.use_selected_features)
                if self.model_type == "random_forest":
                    mlflow.log_param("n_estimators", self.n_estimators)

                # Log metrics
                mlflow.log_metric("val_accuracy", self.val_accuracy)
                mlflow.log_metric("val_f1", self.val_f1)
                mlflow.log_metric("val_auc", self.val_auc)
                mlflow.log_metric("test_accuracy", self.test_accuracy)
                mlflow.log_metric("test_f1", self.test_f1)
                mlflow.log_metric("test_auc", self.test_auc)

                # Save feature names as parameter
                feature_names = list(self.X_train.columns)
                mlflow.log_param("feature_names", str(feature_names))

                # Try to register model (works in newer MLflow versions)
                try:
                    print("Logging model as artifact only first...")
                    mlflow.sklearn.log_model(self.model, "model")
                    print("Successfully logged model as artifact")

                    # Now try to register the model
                    print("Now trying to register model...")
                    model_details = mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/model",
                        "cardio-disease-model",
                    )
                    print(
                        f"Successfully registered model: {model_details.name} version {model_details.version}"
                    )
                except Exception as e:
                    print(f"Error with model registration: {e}")
                    print("Falling back to logging model as artifact only")
        except Exception as e:
            print(f"Error with MLflow tracking: {e}")
            import traceback

            traceback.print_exc()

        self.next(self.end)

    @kubernetes
    @timeout(minutes=5)
    @step
    def end(self):
        """
        End the flow.
        """
        print("Training flow completed!")
        print(f"Selected model: {self.best_model['type']}")
        print(f"Test AUC: {self.test_auc:.4f}")


if __name__ == "__main__":
    CardioTrainingFlow()
