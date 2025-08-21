
#IMPORT ALL LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import joblib

# LOAD DATASET

df = pd.read_csv('medical_data.csv')
df.info()


#Exploratory Data Analysis (EDA)

# df.describe(), df.isna().sum()

# Value counts for categorical columns

# Histograms / boxplots for numerical features

# Correlation heatmap and scatterplots for suspicious pairs

# Investigate class balance (if labeled)



# missingness
missing = df.isna().mean().sort_values(ascending=False)
print(missing[missing>0])

# distributions
df.hist(figsize=(12,10), bins=30);

# correlation heatmap - only numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=False, cmap='vlag', center=0)
plt.title('Correlation matrix (numeric features)')


# DATA Preprocessing

num_features = ['age','hemoglobin','wbc','rbc','platelets','glucose','cholesterol']  # adapt
cat_features = ['gender','smoking_status']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])


#Feature engineering
# Ratios: neutrophil/lymphocyte ratio, hdl/ldl if available

# Flags: is_hypertensive from BP thresholds, anemia_flag from Hb thresholds

# Binned age groups: young/adult/senior

# Composite risk score: weighted sum of key risk markers (expert-driven)


# example thresholds — verify clinically for your population
df['anemia_flag'] = ((df['gender']=='Male') & (df['hemoglobin'] < 13)) | ((df['gender']=='Female') & (df['hemoglobin'] < 12))
df['anemia_flag'] = df['anemia_flag'].astype(int)


#Creating labels


def label_row(r):
    if r['hemoglobin'] < 8 or r['platelets'] < 50000:
        return 3  # urgent: refer to physician
    elif r['hemoglobin'] < 12 or r['glucose']>200:
        return 2  # lifestyle + check-up
    else:
        return 0  # no action needed

df['label'] = df.apply(label_row, axis=1)



#Baseline model & evaluation


from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=['label','recommendation'], errors='ignore')  # adapt
y = df['label']  # or 'recommendation'

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))



#Improve model: imbalance, CV, tuning


from scipy.stats import randint

params = {
    'clf__n_estimators': randint(50,300),
    'clf__max_depth': randint(3,20),
    'clf__min_samples_split': randint(2,10)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rs = RandomizedSearchCV(model_pipeline, params, n_iter=25, cv=cv, scoring='f1_macro', n_jobs=-1)
rs.fit(X_train, y_train)
print(rs.best_params_)



#Interpretability & Explainability



import shap

# get the trained tree model and transformed X_train
clf = rs.best_estimator_.named_steps['clf']
X_train_trans = rs.best_estimator_.named_steps['preprocessor'].transform(X_train)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_train_trans)

# global importance
shap.summary_plot(shap_values, X_train_trans)





#Recommendation logic

label_map = {
    0: "No action needed. Maintain healthy lifestyle and routine check-ups.",
    1: "Regular check-up recommended within 3 months; monitor vitals.",
    2: "Lifestyle changes + consult physician for medication assessment.",
    3: "Urgent: Immediate clinical evaluation advised."
}

def generate_recommendation(raw_patient_row, model, preprocessor):
    from prepare_input_data import prepare_input_data
    
    # Prepare complete input data
    X = prepare_input_data(raw_patient_row)
    
    # Make predictions
    prob = model.predict_proba(X)  # use pipeline
    pred = model.predict(X)[0]
    conf = prob.max()
    explanation = None  # get SHAP explanation for the row
    text = f"{label_map[pred]} (confidence: {conf:.2f})"
    return {'label': pred, 'text': text, 'confidence': float(conf)}



#Save model & deploy


joblib.dump(rs.best_estimator_, 'best_model.joblib')



# app.py
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('best_model.joblib')

st.title("Personalized Healthcare Recommendations Demo")
age = st.number_input("Age", 0, 120, 30)
gender = st.selectbox("Gender", ["Male","Female"])
hemoglobin = st.number_input("Hemoglobin", 0.0, 25.0, 13.5)
# add other inputs...

if st.button("Get recommendation"):
    row = {'age': age, 'gender': gender, 'hemoglobin': hemoglobin}
    out = generate_recommendation(row, model, None)  # adapt to your helper
    st.write(out['text'])



