# src/trainingflow.py
from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class CardioTrainingFlow(FlowSpec):
    """
    A flow to train models for cardiovascular disease prediction.
    """
    
    # Parameters for the flow
    data_path = Parameter('data_path', 
                        default='data/cardio/',
                        help='Path to the cardio data directory')
    
    random_seed = Parameter('random_seed', 
                          default=42,
                          help='Random seed for reproducibility')
    
    n_estimators = Parameter('n_estimators', 
                           default=100,
                           help='Number of estimators for Random Forest')
    
    use_selected_features = Parameter('use_selected_features', 
                                    default=True,
                                    help='Whether to use feature-selected datasets')

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
        self.X_train = self.train_data.drop('cardio', axis=1)
        self.y_train = self.train_data['cardio']
        
        self.X_val = self.val_data.drop('cardio', axis=1)
        self.y_val = self.val_data['cardio']
        
        self.X_test = self.test_data.drop('cardio', axis=1)
        self.y_test = self.test_data['cardio']
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Validation data shape: {self.X_val.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        
        # Go to next steps in parallel
        self.next(self.train_logistic, self.train_rf)
    
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
        self.val_auc = roc_auc_score(self.y_val, self.model.predict_proba(self.X_val)[:, 1])
        
        print(f"Logistic Regression Validation Metrics:")
        print(f"Accuracy: {self.val_accuracy:.4f}")
        print(f"F1 Score: {self.val_f1:.4f}")
        print(f"ROC AUC: {self.val_auc:.4f}")
        
        self.model_type = "logistic_regression"
        self.next(self.choose_model)
    
    @step
    def train_rf(self):
        """
        Train a Random Forest classifier.
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=self.random_seed
        )
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate on validation set
        y_pred = self.model.predict(self.X_val)
        self.val_accuracy = accuracy_score(self.y_val, y_pred)
        self.val_f1 = f1_score(self.y_val, y_pred)
        self.val_auc = roc_auc_score(self.y_val, self.model.predict_proba(self.X_val)[:, 1])
        
        print(f"Random Forest Validation Metrics:")
        print(f"Accuracy: {self.val_accuracy:.4f}")
        print(f"F1 Score: {self.val_f1:.4f}")
        print(f"ROC AUC: {self.val_auc:.4f}")
        
        self.model_type = "random_forest"
        self.next(self.choose_model)
    
    @step
    def choose_model(self, inputs):
        """
        Choose the best model based on validation AUC.
        """
        # Compare models based on validation AUC
        models = [
            {
                "model": inp.model,
                "type": inp.model_type,
                "auc": inp.val_auc,
                "accuracy": inp.val_accuracy,
                "f1": inp.val_f1
            }
            for inp in inputs
        ]
        
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
            
    @step
    def evaluate_model(self):
        """
        Evaluate the best model on the test set.
        """
        # Evaluate on test set
        y_pred = self.model.predict(self.X_test)
        self.test_accuracy = accuracy_score(self.y_test, y_pred)
        self.test_f1 = f1_score(self.y_test, y_pred)
        self.test_auc = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        
        print(f"Test Metrics:")
        print(f"Accuracy: {self.test_accuracy:.4f}")
        print(f"F1 Score: {self.test_f1:.4f}")
        print(f"ROC AUC: {self.test_auc:.4f}")
        
        self.next(self.register_model)
    
    @step
    def register_model(self):
        """
        Register the best model with MLflow.
        """
        import mlflow
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri('http://127.0.0.1:5000')  # Local MLflow server
        mlflow.set_experiment('cardio-disease-metaflow')
        
        # Start MLflow run and log model
        with mlflow.start_run(run_name=f"metaflow-{self.best_model['type']}"):
            # Log parameters
            mlflow.log_param("model_type", self.best_model['type'])
            mlflow.log_param("use_selected_features", self.use_selected_features)
            if self.best_model['type'] == 'random_forest':
                mlflow.log_param("n_estimators", self.n_estimators)
            
            # Log metrics
            mlflow.log_metric("val_accuracy", self.best_model['accuracy'])
            mlflow.log_metric("val_f1", self.best_model['f1'])
            mlflow.log_metric("val_auc", self.best_model['auc'])
            mlflow.log_metric("test_accuracy", self.test_accuracy)
            mlflow.log_metric("test_f1", self.test_f1)
            mlflow.log_metric("test_auc", self.test_auc)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model, 
                "model",
                registered_model_name="cardio-disease-model"
            )
            
            # Save feature names for inference
            feature_names = list(self.X_train.columns)
            mlflow.log_param("feature_names", feature_names)
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow.
        """
        print("Training flow completed!")
        print(f"Selected model: {self.best_model['type']}")
        print(f"Test AUC: {self.test_auc:.4f}")


if __name__ == '__main__':
    CardioTrainingFlow()
