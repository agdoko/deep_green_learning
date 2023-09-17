import ee
import os


def auth_ee():
    service_account = os.environ.get('SERVICE_ACCOUNT')
    print(service_account)
    try:
        print('We did it')
        credentials = ee.ServiceAccountCredentials(service_account, 'ee_auth/semiotic-garden-395711-26d51679d83a.json')

        return ee.Initialize(credentials)
    except KeyError as error:
        print(error)
