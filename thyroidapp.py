from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("thyroid_predictor.pkl")

# Define feature names EXACTLY as they were during training (from your training script)
feature_names = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG"
]


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Collect form data
            patient_data = {}
            for feature in feature_names:
                # Default to 0 if not provided, otherwise convert to float
                value = request.form.get(feature, 0)
                patient_data[feature] = float(value) if value else 0.0

            # Convert to DataFrame with correct feature names
            input_df = pd.DataFrame([patient_data], columns=feature_names)

            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0, 1] * 100

            # Format result
            result = "Healthy" if prediction == 0 else "Thyroid Issue Detected"

            return render_template("index1.html", result=result, probability=probability, error=None)

        except ValueError as e:
            # Handle invalid input (e.g., non-numeric values)
            error_message = f"Error: Please ensure all inputs are valid numbers. Details: {str(e)}"
            return render_template("index1.html", result=None, probability=None, error=error_message)
        except Exception as e:
            # Catch any other unexpected errors
            error_message = f"An unexpected error occurred: {str(e)}"
            return render_template("index1.html", result=None, probability=None, error=error_message)

    # Show form on GET request
    return render_template("index1.html", result=None, probability=None, error=None)


if __name__ == "__main__":
    app.run(debug=True)