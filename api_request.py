import requests
import json
import argparse


def dump_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(
            f"Error: Unable to parse the JSON file '{file_path}'. It might be invalid JSON."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_json():
    parser = argparse.ArgumentParser(description="Dump the contents of a JSON file.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file")
    args = parser.parse_args()

    file_path = args.file_path
    data = dump_json_file(file_path)
    return data


if __name__ == "__main__":
    url = "http://localhost:5000/predict"

    # import the data from the file path passed as an argument
    parser = argparse.ArgumentParser(description="Dump the contents of a JSON file.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file")
    args = parser.parse_args()

    file_path = args.file_path
    data = dump_json_file(file_path)
    input_data = json.loads(data)

    # store the visit_id for later use in file name
    visit_id = input_data["visit_id"]

    # the content type
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}

    # Make the request and display the response
    resp = requests.post(url, input_data, headers=headers)
    print(resp.text)

    # save the response to a file as a json object
    with open(f"./data/predictions/{visit_id}_prediction.json", "w") as f:
        json.dump(resp.json(), f)
