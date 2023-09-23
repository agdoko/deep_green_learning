import ee
import os
from google.cloud import storage
import json
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params

# Initialize the Google Cloud Storage client
client = storage.Client()

# Specify the GCS bucket and JSON file name
json_file_name = 'authentication_keys/semiotic_garden_key.json'

# Get a reference to the JSON file in GCS
bucket = client.get_bucket(params.BUCKET)
blob = bucket.blob(json_file_name)

# Download the JSON file's content as a string
json_content = blob.download_as_text()

# Parse the JSON content into a Python dictionary
json_data = json.loads(json_content)

# Write JSON file
with open('cred.json', 'w') as f:
    json.dump(json_data, f)


#print(json_content)
#print(json_content.keys())
#print(type(json_content))



def auth_ee():
    service_account = os.environ.get('SERVICE_ACCOUNT')
    print(service_account)
    try:
        print('We did it')
        credentials = ee.ServiceAccountCredentials(service_account, "cred.json")

        return ee.Initialize(credentials)
    except KeyError as error:
        print(error)
