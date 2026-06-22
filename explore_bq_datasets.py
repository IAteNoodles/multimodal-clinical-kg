"""Check other projects and look for CXR/ECG-specific datasets."""
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.cloud import bigquery
from pathlib import Path
import os

BQ_TOKEN_PATH = Path(os.environ.get("BQ_TOKEN_PATH", r"C:\Users\Noodl\Projects\Research\Exploration-MJ\data\mimic_5k\token.json"))
BQ_PROJECT_ID = "physionet-data-498016"

def get_bq_client():
    creds = Credentials.from_authorized_user_file(str(BQ_TOKEN_PATH))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(BQ_TOKEN_PATH, 'w') as f:
            f.write(creds.to_json())
    return bigquery.Client(credentials=creds, project=BQ_PROJECT_ID)

def bq_query(client, sql, project=None):
    print(f"\n--- Query (project={project or 'default'}) ---\n{sql}\n---")
    try:
        df = client.query(sql).to_dataframe()
        print(f"Rows: {len(df)}")
        print(df.to_string())
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    client = get_bq_client()

    # 1. Check mimic-498015 project for datasets
    print("="*80)
    print("Datasets in mimic-498015 project")
    print("="*80)
    try:
        mimic_client = bigquery.Client(credentials=client._credentials, project="mimic-498015")
        datasets = list(mimic_client.list_datasets())
        print(f"Found {len(datasets)} datasets:")
        for ds in datasets:
            print(f"  {ds.dataset_id}")
            try:
                tables = list(mimic_client.list_tables(f"mimic-498015.{ds.dataset_id}"))
                for t in tables:
                    print(f"    {t.table_id}")
            except Exception as e:
                print(f"    ERROR listing tables: {e}")
    except Exception as e:
        print(f"ERROR: {e}")

    # 2. Try mimiciv_note dataset in physionet-data
    print("\n" + "="*80)
    print("Try mimiciv_note / mimiciv_3_1_note in physionet-data")
    print("="*80)
    for ds_name in ["mimiciv_note", "mimiciv_3_1_note", "mimiciv_3_1_hosp_note",
                     "mimic_cxr_jpg", "mimic_cxr_2_0_0"]:
        try:
            tables = list(client.list_tables(f"physionet-data.{ds_name}"))
            print(f"\nphysionet-data.{ds_name} ({len(tables)} tables):")
            for t in tables:
                print(f"  {t.table_id}")
        except Exception as e:
            print(f"physionet-data.{ds_name}: NOT FOUND or ERROR")

    # 3. Try querying mimic_cxr tables directly (we know dataset exists but access denied)
    # Try specific known table names
    print("\n" + "="*80)
    print("Try known mimic_cxr table names")
    print("="*80)
    for table in ["record_list", "chexpert", "metadata", "split", "mimic_cxr_record_list",
                   "cxr_record_list", "radiology"]:
        try:
            sql = f"SELECT COUNT(*) as n FROM `physionet-data.mimic_cxr.{table}` LIMIT 1"
            bq_query(client, sql)
        except Exception as e:
            print(f"  mimic_cxr.{table}: DENIED or NOT FOUND")

    # 4. Broader ECG overlap using all ECG procedure codes + chest x-ray procedure code
    print("\n" + "="*80)
    print("Overlap: ECG (8951+8952+8953) + Chest X-ray (8744)")
    print("="*80)
    sql_overlap2 = """
    WITH ecg_patients AS (
        SELECT DISTINCT subject_id
        FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd`
        WHERE icd_code IN ('8951', '8952', '8953')
    ),
    cxr_patients AS (
        SELECT DISTINCT subject_id
        FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd`
        WHERE icd_code IN ('8744', '8749')
    ),
    all_hosp AS (
        SELECT DISTINCT subject_id
        FROM `physionet-data.mimiciv_3_1_hosp.admissions`
    )
    SELECT
        (SELECT COUNT(*) FROM all_hosp) as n_total_hosp_patients,
        (SELECT COUNT(*) FROM ecg_patients) as n_ecg_patients,
        (SELECT COUNT(*) FROM cxr_patients) as n_cxr_proc_patients,
        (SELECT COUNT(*) FROM ecg_patients e INNER JOIN cxr_patients c ON e.subject_id = c.subject_id) as n_overlap
    """
    bq_query(client, sql_overlap2)

    # 5. Check rhythm table in derived (might have ECG rhythm interpretations)
    print("\n" + "="*80)
    print("Check rhythm table (derived)")
    print("="*80)
    sql_rhythm = """
    SELECT * FROM `physionet-data.mimiciv_3_1_derived.rhythm` LIMIT 5
    """
    bq_query(client, sql_rhythm)

    # 6. Count unique patients in rhythm table
    print("\n" + "="*80)
    print("Rhythm table patient count")
    print("="*80)
    sql_rhythm_count = """
    SELECT COUNT(DISTINCT subject_id) as n_patients FROM `physionet-data.mimiciv_3_1_derived.rhythm`
    """
    bq_query(client, sql_rhythm_count)

    # 7. Check cardiac_marker table
    print("\n" + "="*80)
    print("cardiac_marker table")
    print("="*80)
    sql_cm = """
    SELECT * FROM `physionet-data.mimiciv_3_1_derived.cardiac_marker` LIMIT 5
    """
    bq_query(client, sql_cm)

    # 8. Check for MIMIC-IV-ECG dataset - try different project
    print("\n" + "="*80)
    print("Try MIMIC-IV-ECG in mimic-498015 project")
    print("="*80)
    for ds_name in ["mimiciv_ecg", "mimic_iv_ecg", "mimic-iv-ecg", "mimiciv_3_1_ecg",
                     "ecg", "mimiciv_3_1_ecg"]:
        try:
            tables = list(mimic_client.list_tables(f"mimic-498015.{ds_name}"))
            print(f"\nmimic-498015.{ds_name} ({len(tables)} tables):")
            for t in tables:
                print(f"  {t.table_id}")
        except Exception as e:
            print(f"mimic-498015.{ds_name}: NOT FOUND or ERROR")

    # 9. Full overlap analysis: ECG procedure patients + total MIMIC-IV patients
    print("\n" + "="*80)
    print("Comprehensive overlap: ECG proc + MIMIC-IV hosp + MIMIC-IV ICU")
    print("="*80)
    sql_comprehensive = """
    WITH ecg AS (
        SELECT DISTINCT subject_id
        FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd`
        WHERE icd_code IN ('8951', '8952')
    ),
    cxr AS (
        SELECT DISTINCT subject_id
        FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd`
        WHERE icd_code IN ('8744', '8749')
    ),
    hosp AS (
        SELECT DISTINCT subject_id FROM `physionet-data.mimiciv_3_1_hosp.admissions`
    ),
    icu AS (
        SELECT DISTINCT subject_id FROM `physionet-data.mimiciv_3_1_icu.icustays`
    )
    SELECT
        (SELECT COUNT(*) FROM hosp) as n_hosp,
        (SELECT COUNT(*) FROM icu) as n_icu,
        (SELECT COUNT(*) FROM ecg) as n_ecg_proc,
        (SELECT COUNT(*) FROM cxr) as n_cxr_proc,
        (SELECT COUNT(*) FROM ecg INNER JOIN cxr USING(subject_id)) as n_ecg_and_cxr,
        (SELECT COUNT(*) FROM ecg INNER JOIN hosp USING(subject_id)) as n_ecg_in_hosp,
        (SELECT COUNT(*) FROM cxr INNER JOIN hosp USING(subject_id)) as n_cxr_in_hosp,
        (SELECT COUNT(*) FROM ecg INNER JOIN cxr INNER JOIN hosp USING(subject_id)) as n_ecg_cxr_hosp,
        (SELECT COUNT(*) FROM ecg INNER JOIN icu USING(subject_id)) as n_ecg_in_icu,
        (SELECT COUNT(*) FROM cxr INNER JOIN icu USING(subject_id)) as n_cxr_in_icu,
        (SELECT COUNT(*) FROM ecg INNER JOIN cxr INNER JOIN icu USING(subject_id)) as n_ecg_cxr_icu
    """
    bq_query(client, sql_comprehensive)

    print("\n" + "="*80)
    print("DONE")
    print("="*80)

if __name__ == "__main__":
    main()