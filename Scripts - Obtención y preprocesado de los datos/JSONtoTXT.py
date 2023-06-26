import pandas as pd
df = pd.read_json (r'data_stix.json')
df.to_csv (r'data_stix.txt', index = False)