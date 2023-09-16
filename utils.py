import ee
import os
from io import StringIO
def auth_ee():
    service_account = os.environ.get('SERVICE_ACCOUNT')
    credentials = ee.ServiceAccountCredentials(service_account, 'ee_auth/semiotic-garden-395711-26d51679d83a.json')

    try:
        print('We did it')
        return ee.Initialize(credentials)
    except:
        print(':(')
