# src/scoringflow.py
from metaflow import FlowSpec, step, Parameter, Flow, JSONType
import pandas as pd
import numpy as np
import mlflow

class CardioScoringFlow(FlowSpec):
    """
    A flow to score new data using the trained cardiovascular disease prediction model.
    """
    
    # Parameter to accept a single sample or path to a file with samples
    sample_data = Parameter('sample_data',
                           type=JSONType,
                           required=True,
                           help='JSON representation of sample data to score')
    
    model_name = Parameter('model_name',
                          default='cardio-disease-model',
                          help='Name of the registered model to use')
    
    model_stage = Parameter('model_stage',
                           default='latest',
                           help='Stage of the model to use (latest, staging, production)')

    @step
    def start(self):
        """
        Load the trained model and prepare the input data.
        """
        import mlflow
        import pandas as pd
        import json
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri('http://127.0.0.1:5000')  # Local MLflow server
        
        # Load the latest training run to get feature names
        train_run = Flow('CardioTrainingFlow').latest_run
        self.train_run_id = train_run.pathspec
        
        print(f"Using model from training run: {self.train_run_id}")
        
        # Load the model from MLflow
        try:
            self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}/{self.model_stage}")
            print(f"Loaded model: {self.model_name}/{self.model_stage}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Process input data
        self.input_data = self.sample_data
        
        # If input is a file path, load it
        if isinstance(self.input_data, str) and self.input_data.endswith(('.csv', '.json')):
            if self.input_data.endswith('.csv'):
                self.data = pd.read_csv(self.input_data)
            else:  # json file
                with open(self.input_data, 'r') as f:
                    self.data = pd.DataFrame(json.load(f))
        else:
            # Single sample or list of samples passed directly as parameter
            self.data = pd.DataFrame([self.input_data] if isinstance(self.input_data, dict) else self.input_data)
        
        print(f"Input data shape: {self.data.shape}")
        
        self.next(self.preprocess)
        
    @step
    def preprocess(self):
        """
        Preprocess the input data to match the format expected by the model.
        """
        # Get the feature names from the training run
        train_run = Flow('CardioTrainingFlow').latest_run
        feature_names = train_run['register_model'].task.data.X_train.columns.tolist()
        
        # Check if all required features are present
        missing_features = [feat for feat in feature_names if feat not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Make sure only relevant features are used and in correct order
        self.processed_data = self.data[feature_names]
        
        print("Preprocessing completed")
        self.next(self.predict)
    
    @step
    def predict(self):
        """
        Make predictions using the loaded model.
        """
        # Make predictions
        self.predictions = self.model.predict(self.processed_data)
        self.probabilities = self.model.predict_proba(self.processed_data)[:, 1]
        
        # Create a DataFrame with results
        self.results = pd.DataFrame({
            'prediction': self.predictions,
            'probability': self.probabilities
        })
        
        # Add index from original data if available
        if self.data.index.name:
            self.results.index = self.data.index
        
        print("Predictions generated")
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow and return results.
        """
        print("Scoring flow completed!")
        print(f"Number of samples scored: {len(self.results)}")
        print("Results preview:")
        print(self.results.head())
        
        # save the results to a file
        self.results.to_csv('predictions.csv', index=False)


if __name__ == '__main__':
    CardioScoringFlow()
