# Thyroid_predictor
**Thyroid_predictor:** A sleek web tool that predicts thyroid health using a Random Forest model. Enter patient details like age, sex, and lab values (TSH, T3, etc.) to get an instant diagnosis—healthy or at risk—with a risk percentage, all wrapped in a clean, user-friendly interface.

Hey there! Welcome to **ThyroidGuard**, a nifty web tool I crafted to predict thyroid health risks with some smart tech. Picture this: you pop in details like age, sex, or TSH levels, and it spits out whether your thyroid’s chill or sounding the alarm—complete with a risk percentage. Pretty cool, right?

I built this with Python, leaning on scikit-learn’s Random Forest magic for predictions and Flask to spin up a smooth webpage. It’s trained on a solid thyroid dataset, so it knows its stuff!

## What’s Inside?
- **`thyroidapp.py`**: The Flask core—runs the show and taps the trained model for predictions.
- **`thyroidpredictor.py`**: Pre-trained Random Forest weights, ready to roll.
- **`index1.html`**: The webpage—clean inputs and a button to check your health.
- **`style1.css`**: Custom CSS to keep it looking sharp.
- **`images`**: Screenshots of the action.

## How It Works
1. **Training**: I trained a Random Forest model with scikit-learn on a thyroid dataset (think age, sex, TSH, T3, etc.), tuning it to spot health patterns.
2. **Web App**: Run `thyroidapp.py`, and it loads `thyroid_guard_model.pkl`. Enter your data, click "Check Thyroid Health," and get a result—0 (healthy) or 1 (issue)—plus a risk probability.
3. **Output**: The page lights up with “Healthy” (green) or “Thyroid Issue Detected” (red), based on the model’s call.

## Screenshots
Here’s ThyroidGuard in the spotlight:

- **Webpage:**
- ![Screenshot 2025-03-07 193246](https://github.com/user-attachments/assets/b04c534a-23b1-4bf9-ac90-ae454c6b54dd)

- **Model evaluation:**
- ![Screenshot 2025-03-07 193326](https://github.com/user-attachments/assets/2badc072-9b1a-45ab-aca0-ace8cf901d34)

## Running It Yourself
1. **Clone the Repo**:
 . git clone https://github.com/poojith9/thyroid_predictor.git
 . cd thyroid_predictor
2. **Install Dependencies**:
 .pip install flask pandas numpy scikit-learn joblib matplotlib seaborn
3. **Launch It**:
 . python thyroidapp.py 
