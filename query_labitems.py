from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.cloud import bigquery
import json, csv, os

token_path = r"C:\Users\Noodl\Projects\Research\Exploration-MJ\data\mimic_5k\token.json"
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

queries = {
    "Q1_All_lab_items": """
SELECT itemid, label, category, fluid
FROM `physionet-data.mimiciv_3_1_hosp.d_labitems`
WHERE category IN ('Chemistry', 'Hematology', 'Blood Gas')
AND fluid = 'Blood'
ORDER BY category, label
""",
    "Q2_Top_120_frequent_lab_itemids": """
SELECT le.itemid, d.label, d.category, COUNT(*) as cnt
FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
INNER JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` d ON le.itemid = d.itemid
WHERE d.category IN ('Chemistry', 'Hematology', 'Blood Gas')
AND d.fluid = 'Blood'
GROUP BY le.itemid, d.label, d.category
ORDER BY cnt DESC
LIMIT 120
""",
    "Q3_All_lab_items_full": """
SELECT itemid, label, category, fluid
FROM `physionet-data.mimiciv_3_1_hosp.d_labitems`
WHERE category IN ('Chemistry', 'Hematology', 'Blood Gas')
AND fluid = 'Blood'
ORDER BY category, label
"""
}

out_dir = r"C:\Users\Noodl\Projects\Research\MultiModal\bq_results"
os.makedirs(out_dir, exist_ok=True)

for name, sql in queries.items():
    print(f"\n{'='*80}")
    print(f"Running {name}...")
    query_job = client.query(sql)
    rows = list(query_job)
    print(f"Got {len(rows)} rows")
    
    csv_path = os.path.join(out_dir, f"{name}.csv")
    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows([dict(r) for r in rows])
        print(f"Wrote {csv_path}")
    
    print(f"\n--- {name} FULL RESULTS ---")
    for r in rows:
        print(dict(r))

print("\n\nDone. All results saved to bq_results/")