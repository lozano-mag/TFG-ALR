import requests
import json

# API URL
url = 'https://otx.alienvault.com/api/v1/pulses/subscribed?page=1'

# Headers dictionary
headers = {
    'X-OTX-API-KEY': 'Cambiar por API-KEY propia',
}

# Make the GET request
response = requests.get(url, headers=headers)

# Check the response
if response.status_code == 200:
    # Parse JSON data
    data = response.json()

    # Write data to file
    with open('data_stix.json', 'w') as outfile:
        json.dump(data, outfile)
else:
    print(f'Error: {response.status_code}')
