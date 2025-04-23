# src/scoringflowgcp.py
from metaflow import (
    FlowSpec,
    step,
    Parameter,
    Flow,
    JSONType,
    kubernetes,
    conda_base,
    retry,
    timeout,
    catch,
)
import pandas as pd
import numpy as np
import mlflow


@conda_base(
    python="3.11",
    libraries={
        "pandas": "2.2.2",
        "numpy": "1.26.4",
        "scikit-learn": "1.5.1",
        "mlflow": "2.15.1",
        "gcsfs": "2025.3.2",
        "google-cloud-storage": "3.1.0",
    },
)
class CardioScoringFlow(FlowSpec):
    """
    A flow to score new data using the trained cardiovascular disease prediction model.
    """

    # Parameter to accept JSON data directly
    sample_data = Parameter(
        "sample_data",
        type=JSONType,
        required=True,
        help="JSON data to score - this should be a dictionary or list of dictionaries",
    )

    model_name = Parameter(
        "model_name",
        default="cardio-disease-model",
        help="Name of the registered model to use",
    )

    model_stage = Parameter(
        "model_stage",
        default="latest",
        help="Stage of the model to use (latest, staging, production)",
    )

    @kubernetes
    @retry(times=3)
    @timeout(minutes=15)
    @catch(var="load_error", print_exception=True)
    @step
    def start(self):
        """
        Load the trained model and prepare the input data.
        """
        import mlflow
        import pandas as pd
        import json
        import os

        # Try setting username/password for MLflow authentication
        os.environ["MLFLOW_TRACKING_USERNAME"] = "mlflow"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "mlflow"

        # Set up MLflow tracking
        mlflow.set_tracking_uri(
            "https://mlflow-server-824469552969.us-west2.run.app"
        )  # Using GCP MLflow server address

        # Load the latest training run to get feature names
        train_run = Flow("CardioTrainingFlow").latest_run
        self.train_run_id = train_run.pathspec

        print(f"Using model from training run: {self.train_run_id}")

        # Load the model from MLflow
        try:
            self.model = mlflow.sklearn.load_model(
                f"models:/{self.model_name}/{self.model_stage}"
            )
            print(f"Loaded model: {self.model_name}/{self.model_stage}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Process sample_data into DataFrame
        if isinstance(self.sample_data, dict):
            # Single sample
            self.data = pd.DataFrame([self.sample_data])
            print("Processing single sample")
        else:
            # List of samples
            self.data = pd.DataFrame(self.sample_data)
            print(f"Processing {len(self.sample_data)} samples")

        print(f"Input data shape: {self.data.shape}")

        self.next(self.preprocess)

    @kubernetes
    @retry(times=2)
    @timeout(minutes=10)
    @catch(var="preprocessing_error", print_exception=True)
    @step
    def preprocess(self):
        """
        Preprocess the input data to match the format expected by the model.
        """
        import numpy as np

        # Handle possible error from previous step
        if hasattr(self, "load_error") and self.load_error:
            print(f"Error in loading model: {self.load_error}")
            raise RuntimeError(f"Cannot continue due to load error: {self.load_error}")

        # Get the feature names from the training run
        train_run = Flow("CardioTrainingFlow").latest_run
        feature_names = train_run["register_model"].task.data.X_train.columns.tolist()

        print(f"Required features: {feature_names}")
        print(f"Available features: {list(self.data.columns)}")

        # Add derived features
        print("Adding derived features...")

        # Calculate BMI = weight(kg) / height(m)^2
        if "bmi" in feature_names and "bmi" not in self.data.columns:
            if "weight" in self.data.columns and "height" in self.data.columns:
                # Convert height from cm to meters
                height_m = self.data["height"] / 100
                self.data["bmi"] = self.data["weight"] / (height_m**2)
                print("Added BMI feature")
            else:
                raise ValueError("Cannot calculate BMI: 'weight' or 'height' missing")

        # Calculate pulse pressure = systolic - diastolic
        if (
            "pulse_pressure" in feature_names
            and "pulse_pressure" not in self.data.columns
        ):
            if "ap_hi" in self.data.columns and "ap_lo" in self.data.columns:
                self.data["pulse_pressure"] = self.data["ap_hi"] - self.data["ap_lo"]
                print("Added pulse_pressure feature")
            else:
                raise ValueError(
                    "Cannot calculate pulse_pressure: 'ap_hi' or 'ap_lo' missing"
                )

        # Any other required derived features would be added here

        # Check again if all required features are present
        missing_features = [
            feat for feat in feature_names if feat not in self.data.columns
        ]
        if missing_features:
            raise ValueError(f"Still missing required features: {missing_features}")

        # Make sure only relevant features are used and in correct order
        self.processed_data = self.data[feature_names]

        print("Preprocessing completed")
        print(f"Processed data columns: {list(self.processed_data.columns)}")
        self.next(self.predict)

    @kubernetes(cpu=2, memory=4000)
    @retry(times=2)
    @timeout(minutes=15)
    @catch(var="prediction_error", print_exception=True)
    @step
    def predict(self):
        """
        Make predictions using the loaded model.
        """
        # Handle possible errors from previous steps
        if hasattr(self, "preprocessing_error") and self.preprocessing_error:
            print(f"Error in preprocessing: {self.preprocessing_error}")
            raise RuntimeError(
                f"Cannot continue due to preprocessing error: {self.preprocessing_error}"
            )

        # Make predictions
        self.predictions = self.model.predict(self.processed_data)
        self.probabilities = self.model.predict_proba(self.processed_data)[:, 1]

        # Create a DataFrame with results
        self.results = pd.DataFrame(
            {"prediction": self.predictions, "probability": self.probabilities}
        )

        # Add index from original data if available
        if self.data.index.name:
            self.results.index = self.data.index

        print("Predictions generated")
        self.next(self.end)

    @kubernetes
    @timeout(minutes=5)
    @step
    def end(self):
        """
        End the flow and return results.
        """
        # Handle possible errors from previous step
        if hasattr(self, "prediction_error") and self.prediction_error:
            print(f"Error in prediction: {self.prediction_error}")
            print("Flow completed with errors!")
            return

        print("Scoring flow completed!")
        print(f"Number of samples scored: {len(self.results)}")
        print("Results preview:")
        print(self.results.head())

        # Print more detailed results for each sample
        for i, (idx, row) in enumerate(self.results.iterrows()):
            print(f"Sample {i + 1}:")
            print(f"  Probability of cardiovascular disease: {row['probability']:.4f}")
            print(
                f"  Prediction: {'Positive' if row['prediction'] == 1 else 'Negative'}"
            )
            print()

        # We don't save to a file since it wouldn't be accessible from Kubernetes


if __name__ == "__main__":
    CardioScoringFlow()
