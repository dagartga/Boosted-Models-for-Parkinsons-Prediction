from flask import Flask, request, jsonify

import pandas as pd
import joblib
import pickle
from pred_pipeline_user_input_app import get_all_updrs_preds


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Welcome to the Parkinsons Prediction</p>"


@app.route("/test_predict", methods=["GET"])
def test_predict():
    input_data = {
        "visit_month": 60,
        "patient_id": 24343,
        "visit_id": 1,
    }

    input_df = pd.DataFrame(input_data, index=[0])
    # get the predictions
    updrs_preds = get_all_updrs_preds(input_df)

    # convert preds to json
    predictions = updrs_preds.to_json(orient="records")

    return predictions


@app.route("/predict", methods=["POST"])
def predict(payload):
    # convert the json input to a dataframe
    input_df = pd.DataFrame(payload, index=[0])

    # get the predictions
    updrs_preds = get_all_updrs_preds(input_df)

    # convert preds to json
    predictions = updrs_preds.to_json(orient="records")

    return predictions


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
