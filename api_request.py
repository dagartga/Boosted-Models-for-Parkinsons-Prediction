import requests
import json

url = "http://localhost:5000/predict"

data = json.dumps({"visit_month": "12", "patient_id": "24343", "visit_id": "24343_12"})

# store the visit_id for later use in file name
visit_id = data["visit_id"]

# the content type
headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}

# Make the request and display the response
resp = requests.post(url, data, headers=headers)
print(resp.text)

# save the response to a file as a json object
with open(f"./data/predictions/{visit_id}_prediction.json", "w") as f:
    json.dump(resp.json(), f)
