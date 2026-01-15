from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)



df = pd.read_csv('crop_recommendation (1).csv')

X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['crop']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    new_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=['N','P','K','temperature','humidity','ph','rainfall']
    )

    new_data_scaled = scaler.transform(new_data)

    probs = model.predict_proba(new_data_scaled)[0]
    classes = model.classes_

    top5_idx = np.argsort(probs)[-5:][::-1]

    result = [(classes[i], probs[i]) for i in top5_idx]

    return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
