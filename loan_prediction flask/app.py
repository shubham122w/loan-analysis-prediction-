from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetching inputs from the form
        income = float(request.form['income'])
        credit_score = float(request.form['credit_score'])
        employment_status = int(request.form['employment_status'])  # 0,1,2
        loan_amount = float(request.form['loan_amount'])
        loan_term = int(request.form['loan_term'])  # 1 or 2

        # Combine all features in correct order
        features = np.array([[income, credit_score, employment_status, loan_amount, loan_term]])

        # Scale features
        scaled_features = scaler.transform(features)

        # Predict using model
        prediction = model.predict(scaled_features)[0]

        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Something went wrong: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
