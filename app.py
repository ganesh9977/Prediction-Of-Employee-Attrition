from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the trained model
knn_classifier = load('knn_classifier.joblib')

# Load the dataset
data = pd.read_csv('employee_attrition_dataset.csv')

# Data preprocessing
data = data.drop(columns=['Employee ID', 'Date of Joining', 'Performance Metrics'])
data.fillna(0, inplace=True)
data = pd.get_dummies(data, drop_first=True)

# Define features (X)
X = data.drop(columns=['Target'])

@app.route('/')
def home():
    prediction = request.args.get('prediction')
    return render_template('index.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    department = request.form['department']
    position = request.form['position']
    tenure = float(request.form['tenure'])
    satisfaction = float(request.form['satisfaction'])
    engagement = float(request.form['engagement'])
    salary = float(request.form['salary'])

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Department': [department],
        'Position': [position],
        'Tenure (years)': [tenure],
        'Employee Satisfaction (out of 10)': [satisfaction],
        'Engagement Score (out of 10)': [engagement],
        'Salary': [salary]
    })

    # Encode categorical variables
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Ensure input features match the model's expected input
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    input_data = input_data[X.columns]

    # Make prediction
    prediction = knn_classifier.predict(input_data)

    # Translate prediction to human-readable form
    predicted_class = 'Stay' if prediction == 0 else 'Resign'

    # Redirect back to the home page with the prediction result as a query parameter
    return redirect(url_for('home', prediction=predicted_class))

if __name__ == '__main__':
    app.run(debug=True)
