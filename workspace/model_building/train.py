# for data manipulation
import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from workspace.model_building.preprocessor import Preprocessor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
# for model training, tuning, and evaluation

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow


#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("predictive_maint_experiment")

api = HfApi()


Xtrain_path = "hf://datasets/chaitanya-yadav/vehicle-predictive-maintenance/Xtrain.csv"
Xtest_path = "hf://datasets/chaitanya-yadav/vehicle-predictive-maintenance/Xtest.csv"
ytrain_path = "hf://datasets/chaitanya-yadav/vehicle-predictive-maintenance/ytrain.csv"
ytest_path = "hf://datasets/chaitanya-yadav/vehicle-predictive-maintenance/ytest.csv"



Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

Xtrain.columns = Xtrain.columns.str.strip()
Xtest.columns = Xtest.columns.str.strip()

# List of numerical features in the dataset
# Define the target variable for the classification task
target = 'Engine Condition'        # Target variable  (0: No, 1: Yes).

# List of numerical features in the dataset
features = [
    'Engine rpm',              # Age of the customer.
    'Lub oil pressure',                # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    'Fuel pressure',         # Duration of the sales pitch delivered to the customer.
    'Coolant pressure',  # Total number of people accompanying the customer on the trip.
    'lub oil temp',       # Total number of follow-ups by the salesperson after the sales pitch.-
    'Coolant temp',   # Preferred hotel rating by the customer.
]


# Set the clas weight to handle class imbalance

#preprocessor = Preprocessor()


#train_df = pd.concat([Xtrain, ytrain], axis=1)
#test_df = pd.concat([Xtest, ytest], axis=1)

#Xtrain_processed, ytrain = preprocessor.fit_transform(train_df, target)
#Xtest_processed, ytest = preprocessor.transform(test_df, target)

# Define base GBM model
#X_train_processed = preprocessor.transform(Xtrain)


#model_pca_gbm = GradientBoostingClassifier(random_state=42)
#model_pipeline = make_pipeline(preprocessor, GradientBoostingClassifier(random_state=42))
pipeline = Pipeline([
    ("preprocessor", Preprocessor()),
    ("model", GradientBoostingClassifier(random_state=42))
])

# Define hyperparameter grid
param_grid = {
    "model__n_estimators": np.arange(125, 200, 25),
    "model__learning_rate": [0.01, 0.03, 0.05, 0.06, 0.2],
    "model__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "model__max_features": [0.3, 0.5, 0.7, 0.9],
}

# Model pipeline


#model_pipeline = make_pipeline(preprocessor, model_pca_gbm)
scorer = metrics.make_scorer(metrics.f1_score, average='weighted')

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    #randomized_cv = RandomizedSearchCV(estimator=model_pca_gbm, param_distributions=param_grid, n_iter=80, scoring=scorer, cv=5, random_state=42, n_jobs = -1)
    randomized_cv = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=80,
    scoring=scorer,
    cv=5,
    random_state=42,
    n_jobs=-1
)
    randomized_cv.fit(Xtrain, ytrain.values.ravel())
    #randomized_cv.fit(Xtrain_processed, ytrain)

    # Log all parameter combinations and their mean test scores
    results = randomized_cv.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(randomized_cv.best_params_)

    # Store and evaluate the best model
    best_model = randomized_cv.best_estimator_

    classification_threshold = 0.6

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_predictive_maintenance_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # create full pipeline manually
    
    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "chaitanya-yadav/vehicle-predictive-maintenance"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_predictive_maintenance_model_v1.joblib",
        path_in_repo="best_predictive_maintenance_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
