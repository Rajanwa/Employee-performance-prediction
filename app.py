from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    input_data = [float(x) for x in request.form.values()]
    input_array = np.array(input_data).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_array)[0]
    
    return render_template('index.html', prediction_text=f'Predicted value: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
