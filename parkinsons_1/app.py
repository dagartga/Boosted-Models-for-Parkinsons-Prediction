from flask import Flask, request, jsonify

import pandas as pd
import joblib
import pickle


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Welcome to the Parkinsons Prediction</p>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
