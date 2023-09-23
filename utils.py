import ee
import os
from google.cloud import storage
import json
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params





def auth_ee(email, json_file):
    #service_account = os.environ.get('SERVICE_ACCOUNT')
    #print(service_account)
    try:
        credentials = ee.ServiceAccountCredentials(email=email, key_data=json_file)
        print('We did it')
        return ee.Initialize(credentials)
    except KeyError as error:
        print(error)
