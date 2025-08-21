import pandas as pd
import numpy as np

def prepare_input_data(input_dict):
    """
    Prepare complete input data for model prediction by filling missing columns
    with appropriate default values based on the expected structure.
    
    Args:
        input_dict (dict): Dictionary with user input from Streamlit
        
    Returns:
        pd.DataFrame: Complete dataframe with all required columns
    """
    # Expected columns based on the training data
    expected_columns = [
        'age', 'gender', 'hemoglobin', 'wbc', 'rbc', 'platelets', 
        'glucose', 'cholesterol', 'smoking_status'
    ]
    
    # Create a complete dictionary with all expected columns
    complete_data = {}
    
    # Fill provided values
    for col in expected_columns:
        if col in input_dict:
            complete_data[col] = input_dict[col]
        else:
            # Set appropriate defaults for missing columns
            if col == 'age':
                complete_data[col] = 40  # Default age
            elif col == 'gender':
                complete_data[col] = 'Male'  # Default gender
            elif col == 'hemoglobin':
                complete_data[col] = 13.0  # Normal hemoglobin
            elif col == 'wbc':
                complete_data[col] = 7.0  # Normal WBC count
            elif col == 'rbc':
                complete_data[col] = 4.5  # Normal RBC count
            elif col == 'platelets':
                complete_data[col] = 250000  # Normal platelet count
            elif col == 'glucose':
                complete_data[col] = 95  # Normal glucose
            elif col == 'cholesterol':
                complete_data[col] = 180  # Normal cholesterol
            elif col == 'smoking_status':
                complete_data[col] = 'Non-smoker'  # Default smoking status
    
    # Convert to DataFrame
    df = pd.DataFrame([complete_data])
    
    # Ensure correct data types
    numeric_cols = ['age', 'hemoglobin', 'wbc', 'rbc', 'platelets', 'glucose', 'cholesterol']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Test the function
if __name__ == "__main__":
    test_input = {
        'age': 30,
        'gender': 'Male',
        'hemoglobin': 13.5
    }
    
    result = prepare_input_data(test_input)
    print("Test result:")
    print(result)
    print("Columns:", result.columns.tolist())
