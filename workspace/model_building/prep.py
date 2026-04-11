# for data manipulation
import pandas as pd
import sklearn
from workspace.model_building.preprocessor import Preprocessor
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/chaitanya-yadav/vehicle-predictive-maintenance/engine_data.csv"
pred_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")



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

# Define predictor matrix (X) using selected numeric and categorical features
X = pred_dataset[features]

# Define target variable
y = pred_dataset[target]

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.3,     # 30% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

#preprocessor = Preprocessor()

#train_df = pd.concat([Xtrain, ytrain], axis=1)
#test_df = pd.concat([Xtest, ytest], axis=1)

#Xtrain_processed, ytrain = preprocessor.fit_transform(train_df, target)
#Xtest_processed, ytest = preprocessor.transform(test_df, target)

version = "v1"
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="chaitanya-yadav/vehicle-predictive-maintenance",
        repo_type="dataset",
    )
