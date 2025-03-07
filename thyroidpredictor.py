import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Load the CSV dataset
data = pd.read_csv("thyroidDF.csv")
print("Dataset loaded. First 5 rows:")
print(data.head())

# Step 2: Simplify the target (0 = normal, 1 = abnormal)
data["thyroid_status"] = data["target"].apply(lambda x: 0 if x == "-" else 1)
data = data.drop("target", axis=1)

# Step 3: Preprocess the data
# Encode t/f columns to 1/0
bool_cols = ["on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds", "sick",
             "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
             "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary",
             "psych", "TSH_measured", "T3_measured", "TT4_measured", "T4U_measured",
             "FTI_measured", "TBG_measured"]
for col in bool_cols:
    data[col] = data[col].map({"t": 1, "f": 0})

# Encode sex (F=0, M=1, missing=-1)
data["sex"] = data["sex"].map({"F": 0, "M": 1}).fillna(-1)

# Convert numeric columns and fill missing values with median (fixed for pandas 3.0 compatibility)
numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    # Assign directly instead of using inplace=True
    data[col] = data[col].fillna(data[col].median())

# Drop irrelevant columns
data = data.drop(["patient_id", "referral_source"], axis=1)

# Step 4: Split features (X) and target (y)
X = data.drop("thyroid_status", axis=1)
y = data["thyroid_status"]

# Step 5: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as 'confusion_matrix.png'.")

# ROC-AUC Score and Curve
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.2f}")

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.savefig("roc_curve.png")
print("ROC curve saved as 'roc_curve.png'.")

# Feature Importance
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
feature_importance = feature_importance.sort_values("Importance", ascending=False)
print("\nTop 10 Feature Importances:")
print(feature_importance.head(10))

plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10))
plt.title("Top 10 Feature Importances")
plt.savefig("feature_importance.png")
print("Feature importance plot saved as 'feature_importance.png'.")

# Step 8: Save the model
joblib.dump(model, "thyroid_predictor.pkl")
print("Model saved as 'thyroid_predictor.pkl'.")

# Step 9: Predict on new data (example)
new_patient = pd.DataFrame({
    "age": [40], "sex": [0], "on_thyroxine": [0], "query_on_thyroxine": [0],
    "on_antithyroid_meds": [0], "sick": [0], "pregnant": [0], "thyroid_surgery": [0],
    "I131_treatment": [0], "query_hypothyroid": [0], "query_hyperthyroid": [0],
    "lithium": [0], "goitre": [0], "tumor": [0], "hypopituitary": [0], "psych": [0],
    "TSH_measured": [1], "TSH": [1.2], "T3_measured": [1], "T3": [2.3],
    "TT4_measured": [1], "TT4": [104], "T4U_measured": [1], "T4U": [1.08],
    "FTI_measured": [1], "FTI": [96], "TBG_measured": [0], "TBG": [28]
})

prediction = model.predict(new_patient)
prob = model.predict_proba(new_patient)[:, 1]
print(f"\nNew Patient Prediction: {prediction[0]} (0 = normal, 1 = abnormal)")
print(f"Risk Probability: {prob[0] * 100:.2f}%")

# Step 10: Save test predictions to CSV
X_test_with_pred = X_test.copy()
X_test_with_pred["actual_thyroid_status"] = y_test
X_test_with_pred["predicted_thyroid_status"] = y_pred
X_test_with_pred.to_csv("thyroid_predictions.csv", index=False)
print("Test predictions saved to 'thyroid_predictions.csv'.")