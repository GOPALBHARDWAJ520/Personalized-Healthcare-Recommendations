import joblib
import pandas as pd
from prepare_input_data import prepare_input_data

# Test the fix
def test_recommendation():
    # Load the model
    try:
        model = joblib.load('best_model.joblib')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test with minimal input (like from Streamlit)
    test_input = {
        'age': 30,
        'gender': 'Male',
        'hemoglobin': 13.5
    }
    
    print("Original input:", test_input)
    
    # Prepare complete data
    complete_data = prepare_input_data(test_input)
    print("Complete data shape:", complete_data.shape)
    print("Complete data columns:", complete_data.columns.tolist())
    print("Complete data values:")
    print(complete_data.iloc[0])
    
    # Test prediction
    try:
        prediction = model.predict(complete_data)
        probability = model.predict_proba(complete_data)
        print("Prediction successful!")
        print("Predicted class:", prediction[0])
        print("Probabilities:", probability[0])
        
        # Map to recommendation
        label_map = {
            0: "No action needed. Maintain healthy lifestyle and routine check-ups.",
            1: "Regular check-up recommended within 3 months; monitor vitals.",
            2: "Lifestyle changes + consult physician for medication assessment.",
            3: "Urgent: Immediate clinical evaluation advised."
        }
        print("Recommendation:", label_map[prediction[0]])
        
    except Exception as e:
        print(f"Prediction error: {e}")

if __name__ == "__main__":
    test_recommendation()
