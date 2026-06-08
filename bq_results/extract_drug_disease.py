import csv
import sys
from collections import Counter, defaultdict

KEYWORDS = ['treat', 'prevent', 'contraindica', 'indicat', 'therapy', 'therapeutic', 'disease', 'pharmacol', 'drug_class', 'may_', 'manages']

filepath = r"C:\Users\Noodl\Projects\Research\Exploration-MJ\data\umls-2026AA-mrrel\MRREL.RRF"
outfile = r"C:\Users\Noodl\Projects\Research\MultiModal\bq_results\mrrel_drug_disease_pairs.csv"
summary_file = r"C:\Users\Noodl\Projects\Research\MultiModal\bq_results\mrrel_drug_disease_summary.txt"

rela_counter = Counter()
sab_counter = Counter()
rela_sab_counter = Counter()
cui1_set = set()
cui2_set = set()
pair_set = set()
cui1_degree = Counter()
rela_samples = defaultdict(list)

with open(filepath, 'r', encoding='utf-8', errors='replace') as fin, \
     open(outfile, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    writer.writerow(['CUI1','STYPE1','REL','CUI2','STYPE2','RELA','SAB'])
    count = 0
    for i, line in enumerate(fin):
        parts = line.strip().split('|')
        if len(parts) > 10:
            rela = parts[7].lower()
            if any(kw in rela for kw in KEYWORDS):
                row = [parts[0], parts[2], parts[3], parts[4], parts[6], parts[7], parts[10]]
                writer.writerow(row)
                count += 1

                rela_counter[parts[7]] += 1
                sab_counter[parts[10]] += 1
                rela_sab_counter[(parts[7], parts[10])] += 1
                cui1_set.add(parts[0])
                cui2_set.add(parts[4])
                pair_set.add((parts[0], parts[4]))
                cui1_degree[parts[0]] += 1

                rela_val = parts[7]
                if len(rela_samples[rela_val]) < 20:
                    rela_samples[rela_val].append(row)

        if (i+1) % 5000000 == 0:
            print(f"Processed {i+1:,} lines, found {count} matches...")

    print(f"Done. Total matches: {count}")

# Write summary
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("UMLS MRREL Drug-Disease Extraction Summary\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total matching lines: {count}\n\n")

    f.write("-" * 60 + "\n")
    f.write("Breakdown by RELA value:\n")
    f.write("-" * 60 + "\n")
    for rela, cnt in rela_counter.most_common():
        f.write(f"  {rela}: {cnt:,}\n")
    f.write("\n")

    f.write("-" * 60 + "\n")
    f.write("Breakdown by SAB source:\n")
    f.write("-" * 60 + "\n")
    for sab, cnt in sab_counter.most_common():
        f.write(f"  {sab}: {cnt:,}\n")
    f.write("\n")

    f.write("-" * 60 + "\n")
    f.write("Breakdown by (RELA, SAB):\n")
    f.write("-" * 60 + "\n")
    for (rela, sab), cnt in rela_sab_counter.most_common(50):
        f.write(f"  ({rela}, {sab}): {cnt:,}\n")
    f.write("\n")

    f.write("-" * 60 + "\n")
    f.write("Quick Analysis:\n")
    f.write("-" * 60 + "\n")
    f.write(f"  Unique CUI1 values (potential drugs): {len(cui1_set):,}\n")
    f.write(f"  Unique CUI2 values (potential diseases): {len(cui2_set):,}\n")
    f.write(f"  Unique (CUI1, CUI2) pairs: {len(pair_set):,}\n")
    f.write("\n")

    f.write("-" * 60 + "\n")
    f.write("Top 10 most-connected drugs (CUI1 with most disease connections):\n")
    f.write("-" * 60 + "\n")
    for cui, deg in cui1_degree.most_common(10):
        f.write(f"  {cui}: {deg} connections\n")
    f.write("\n")

    f.write("-" * 60 + "\n")
    f.write("Sample rows by RELA type (up to 20 each):\n")
    f.write("-" * 60 + "\n")
    for rela in sorted(rela_samples.keys()):
        f.write(f"\n  RELA: {rela} ({rela_counter[rela]:,} total)\n")
        for row in rela_samples[rela]:
            f.write(f"    {row}\n")

print(f"Summary written to {summary_file}")