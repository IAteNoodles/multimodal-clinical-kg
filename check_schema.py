from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.cloud import bigquery
import json, os

token_path = os.environ.get("BQ_TOKEN_PATH", r"C:\Users\Noodl\Projects\Research\Exploration-MJ\data\mimic_5k\token.json")
project_id = "physionet-data-498016"

with open(token_path, 'r') as f:
    token_data = json.load(f)
creds = Credentials.from_authorized_user_info(token_data)
if creds.expired and creds.refresh_token:
    creds.refresh(Request())
    with open(token_path, 'w') as f:
        json.dump(json.loads(creds.to_json()), f)
    print("Token refreshed")

client = bigquery.Client(credentials=creds, project=project_id)

sql = """
SELECT column_name, data_type
FROM `physionet-data.mimiciv_3_1_hosp.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'd_labitems'
ORDER BY ordinal_position
"""
for row in client.query(sql):
    print(f"{row.column_name}: {row.data_type}")