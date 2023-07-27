import os
import pandas as pd
from flask import Flask, request, jsonify

# Assuming you have the prediction function defined in another module
# Replace 'your_module' and 'prediction' with the appropriate names
from your_module import prediction

app = Flask(__name__)

# Function to process the CSV file and get the prediction
def process_csv_and_get_prediction(csv_file):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)

    # Call your prediction function with the DataFrame and get the result
    result = prediction(df)

    return result

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if a file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is in CSV format
    if file.filename.endswith('.csv'):
        # Save the file in a temporary location
        temp_file_path = 'temp.csv'
        file.save(temp_file_path)

        # Process the CSV file and get the prediction
        prediction_result = process_csv_and_get_prediction(temp_file_path)

        # Remove the temporary file
        os.remove(temp_file_path)

        # Return the prediction result as a JSON object
        return jsonify({'prediction': prediction_result})
    else:
        return jsonify({'error': 'Invalid file format. Only CSV files are supported.'})


if __name__ == '__main__':
    app.run(debug=True)
