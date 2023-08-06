import requests
import json

url = "http://localhost:5000/predict"

data = json.dumps({"visit_month": "12", "patient_id": "24343", "visit_id": "24343_12"})

# the content type
headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}

# Make the request and display the response
resp = requests.post(url, data, headers=headers)
print(resp.text)
