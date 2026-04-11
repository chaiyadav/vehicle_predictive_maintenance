import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
#from workspace.model_building.preprocessor import Preprocessor
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="chaitanya-yadav/vehicle-predictive-maintenance", filename="best_predictive_maintenance_model_v1.joblib")

# Load the model
pipeline = joblib.load(model_path)

#preprocessor = artifact["preprocessor"]
#model = artifact["model"]
#st.write("Preprocessor loaded:", hasattr(preprocessor, "features")) 
# Streamlit UI for Vehicle Maintenance Prediction
st.title("PREDICTIVE MAINTENANCE FOR ENGINE HEALTH")
st.write(
    "AI-powered system to predict potential engine failures "
    "based on real-time sensor inputs."
)
# Collect user input
engine_rpm = st.slider("Engine rpm", 61.0, 2239.0, (100.0))
lub_oil_pressure = st.slider("Lub oil pressure", 0.003384, 7.265566, (1.0))
fuel_pressure = st.slider("Fuel pressure", 0.003187, 21.138326, (2.0))
coolant_pressure = st.slider("Coolant pressure", 0.002483, 7.478505, ( 4.0))
lub_oil_temp = st.slider("Lub oil temp", 71.321974, 89.580796, (75.0))
coolant_temp = st.slider("Coolant temp", 61.673325, 195.527912, (80.0))



# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp,
}])

# IMPORTANT: input must be DataFrame
#processed_input, _ = preprocessor.transform(input_data, target_col="Engine Condition")
#processed_input = preprocessor.transform(input_data)
#prediction = model.predict(processed_input)
#prediction_proba = model.predict_proba(processed_input)

# Set the classification threshold
classification_threshold = 0.6

# Predict button
if st.button("Predict"):
    try:
      #processed_input = preprocessor.transform(input_data)
      prediction_proba = pipeline.predict_proba(input_data)[0, 1]
      #prediction_proba = model.predict_proba(processed_input)[0, 1]
      prediction = (prediction_proba >= classification_threshold).astype(int)
      result = "⚠️ Engine failure likely – immediate inspection recommended" if prediction == 1 else "✅ Engine operating normally – no immediate issues detected"
      st.write(f"Based on the information provided, the {result}.")
      st.write(f"Prediction confidence: {prediction_proba:.2f}")
    except Exception as e:
      st.error(f"Error during prediction: {e}")
