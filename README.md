# Diabetes-Prediction-ML
Diabetes prediction using Machine Learning and Python

# ============================================================
# DIABETES PREDICTION USING MACHINE LEARNING
# Algorithm: Logistic Regression
# ============================================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. LOAD DATASET
# ============================================================

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome'
]

df = pd.read_csv(url, names=columns)

print("\n========== DIABETES DATASET ==========")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ============================================================
# 3. DATA VISUALIZATION
# ============================================================

# Diabetes distribution

plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = Non-Diabetic, 1 = Diabetic)")
plt.ylabel("Number of Patients")
plt.show()


# Correlation heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

invalid_columns = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

# Replace invalid zero values with NaN
df[invalid_columns] = df[invalid_columns].replace(0, np.nan)

# Replace missing values with median
for column in invalid_columns:
    df[column] = df[column].fillna(df[column].median())

print("\nData preprocessing completed!")

print("\nMissing values after preprocessing:")
print(df.isnull().sum())


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop('Outcome', axis=1)
y = df['Outcome']

print("\nInput Features:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)


# ============================================================
# 7. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")


# ============================================================
# 8. TRAIN MACHINE LEARNING MODEL
# ============================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print("\nModel training completed successfully!")


# ============================================================
# 9. MODEL PREDICTION
# ============================================================

y_pred = model.predict(X_test_scaled)


# ============================================================
# 10. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Non-Diabetic', 'Diabetic'],
    yticklabels=['Non-Diabetic', 'Diabetic']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test,
    y_pred,
    target_names=['Non-Diabetic', 'Diabetic']
))


# ============================================================
# 13. ACTUAL VS PREDICTED
# ============================================================

comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(comparison.head(20))


# ============================================================
# 14. PREDICTION FOR A SAMPLE PATIENT
# ============================================================

patient_data = np.array([[
    2,       # Pregnancies
    120,     # Glucose
    70,      # Blood Pressure
    25,      # Skin Thickness
    80,      # Insulin
    28.5,    # BMI
    0.45,    # Diabetes Pedigree Function
    30       # Age
]])

patient_scaled = scaler.transform(patient_data)

prediction = model.predict(patient_scaled)

probability = model.predict_proba(patient_scaled)

print("\n========== SAMPLE PATIENT PREDICTION ==========")

if prediction[0] == 1:
    print("Prediction: Higher predicted likelihood of diabetes")
else:
    print("Prediction: Lower predicted likelihood of diabetes")

print("Diabetes Probability:", round(probability[0][1] * 100, 2), "%")


# ============================================================
# 15. INTERACTIVE PATIENT PREDICTION
# ============================================================

print("\n\n========== DIABETES PREDICTION SYSTEM ==========")

try:
    pregnancies = float(input("Enter number of pregnancies: "))
    glucose = float(input("Enter glucose level: "))
    bp = float(input("Enter blood pressure: "))
    skin = float(input("Enter skin thickness: "))
    insulin = float(input("Enter insulin level: "))
    bmi = float(input("Enter BMI: "))
    dpf = float(input("Enter diabetes pedigree function: "))
    age = float(input("Enter age: "))

    new_patient = np.array([[
        pregnancies,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    new_patient_scaled = scaler.transform(new_patient)

    result = model.predict(new_patient_scaled)[0]

    probabilities = model.predict_proba(new_patient_scaled)[0]

    print("\n========== PREDICTION RESULT ==========")

    if result == 1:
        print("The model predicts a higher likelihood of diabetes.")
    else:
        print("The model predicts a lower likelihood of diabetes.")

    print("Probability of Non-Diabetic:", round(probabilities[0] * 100, 2), "%")
    print("Probability of Diabetic:", round(probabilities[1] * 100, 2), "%")

except ValueError:
    print("Invalid input! Please enter numerical values.")

print("\n========== PROGRAM COMPLETED ==========")
