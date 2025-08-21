import streamlit as st
import pandas as pd
import joblib
from prepare_input_data import prepare_input_data

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('best_model.joblib')

model = load_model()

st.title("🏥 Personalized Healthcare Recommendations")
st.markdown("Get personalized health recommendations based on your medical parameters")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.header("Basic Information")
    age = st.number_input("Age", min_value=0, max_value=120, value=30, help="Your age in years")
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    st.header("Blood Count")
    hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=13.5, 
                                help="Normal range: Men 13.5-17.5, Women 12.0-15.5")
    wbc = st.number_input("White Blood Cells (×10³/μL)", min_value=0.0, max_value=50.0, value=7.0,
                         help="Normal range: 4.0-11.0")
    rbc = st.number_input("Red Blood Cells (×10⁶/μL)", min_value=0.0, max_value=10.0, value=4.5,
                         help="Normal range: Men 4.7-6.1, Women 4.2-5.4")
    platelets = st.number_input("Platelets (×10³/μL)", min_value=0, max_value=1000000, value=250000,
                             help="Normal range: 150,000-450,000")

with col2:
    st.header("Metabolic Panel")
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=500, value=95,
                           help="Normal range: 70-100 mg/dL (fasting)")
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=500, value=180,
                               help="Normal range: <200 mg/dL")
    
    st.header("Lifestyle")
    smoking_status = st.selectbox("Smoking Status", 
                                  ["Non-smoker", "Former", "Current"],
                                  help="Your current smoking status")

# Create input dictionary
input_data = {
    'age': age,
    'gender': gender,
    'hemoglobin': hemoglobin,
    'wbc': wbc,
    'rbc': rbc,
    'platelets': platelets,
    'glucose': glucose,
    'cholesterol': cholesterol,
    'smoking_status': smoking_status
}

# Get recommendation button
if st.button("🔍 Get Health Recommendation", type="primary"):
    try:
        # Prepare complete input data
        X = prepare_input_data(input_data)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        confidence = max(probability)
        
        # Display results
        st.divider()
        st.header("📊 Results")
        
        # Create columns for results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Risk Level", f"Level {prediction}", 
                     delta=f"{confidence:.1%} confidence")
        
        with res_col2:
            # Map prediction to recommendation text
            label_map = {
                0: "No action needed. Maintain healthy lifestyle and routine check-ups.",
                1: "Regular check-up recommended within 3 months; monitor vitals.",
                2: "Lifestyle changes + consult physician for medication assessment.",
                3: "Urgent: Immediate clinical evaluation advised."
            }
            recommendation_text = label_map[prediction]
            st.info(recommendation_text)
        
        # Show probability distribution
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            'Risk Level': [0, 1, 2, 3],
            'Probability': probability
        })
        st.bar_chart(prob_df.set_index('Risk Level'))
        
        # Show input summary
        with st.expander("📋 View Input Summary"):
            st.json(input_data)
            
    except Exception as e:
        st.error(f"Error generating recommendation: {str(e)}")
        st.info("Please check your input values and try again.")

# Footer
st.divider()
st.caption("⚠️ This tool provides general health recommendations based on common medical parameters. Always consult with a healthcare professional for personalized medical advice.")
