import csv
import time
from collections import Counter

filepath = r"C:\Users\Noodl\Projects\Research\Exploration-MJ\data\umls-2026AA-mrrel\MRREL.RRF"
outpath = r"C:\Users\Noodl\Projects\Research\MultiModal\bq_results\mrrel_rela_counts.csv"

keywords = ['treat', 'prevent', 'contraindicated', 'indicat', 'therapy', 'therapeutic', 'disease', 'drug', 'medication', 'pharmacol']

rela_counter = Counter()
rela_sab_counter = Counter()
total_lines = 0
lines_with_rela = 0

start = time.time()

with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        total_lines = i + 1
        parts = line.strip().split('|')
        if len(parts) > 10:
            rela = parts[7]
            sab = parts[10]
            if rela:
                rela_counter[rela] += 1
                rela_sab_counter[(rela, sab)] += 1
                lines_with_rela += 1
        if (i + 1) % 5000000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"Processed {i+1:,} lines... ({rate:,.0f} lines/s, {elapsed:.1f}s)")

elapsed = time.time() - start
print(f"\n{'='*80}")
print(f"DONE. Total lines: {total_lines:,} | Lines with RELA: {lines_with_rela:,} | Unique RELA: {len(rela_counter):,} | Unique (RELA,SAB): {len(rela_sab_counter):,}")
print(f"Time: {elapsed:.1f}s ({total_lines/elapsed:,.0f} lines/s)")

print(f"\n{'='*80}")
print("TOP 50 RELA VALUES BY COUNT:")
print(f"{'='*80}")
print(f"{'Rank':<6} {'RELA':<60} {'Count':>12}")
print('-'*80)
for rank, (rela, cnt) in enumerate(rela_counter.most_common(50), 1):
    print(f"{rank:<6} {rela:<60} {cnt:>12,}")

print(f"\n{'='*80}")
print("TOP 100 RELA VALUES BY COUNT:")
print(f"{'='*80}")
for rank, (rela, cnt) in enumerate(rela_counter.most_common(100), 1):
    print(f"{rank:<6} {rela:<60} {cnt:>12,}")

print(f"\n{'='*80}")
print("TREATMENT/DISEASE/DRUG RELATED (RELA, SAB) COMBINATIONS:")
print(f"{'='*80}")

kw_matches = []
for (rela, sab), cnt in rela_sab_counter.items():
    rela_lower = rela.lower()
    if any(kw in rela_lower for kw in keywords):
        kw_matches.append((rela, sab, cnt))

kw_matches.sort(key=lambda x: -x[2])

print(f"Found {len(kw_matches)} matching (RELA, SAB) combinations")
print(f"{'RELA':<60} {'SAB':<15} {'Count':>12}")
print('-'*90)
for rela, sab, cnt in kw_matches:
    print(f"{rela:<60} {sab:<15} {cnt:>12,}")

# Save full results to CSV
print(f"\nSaving to {outpath}...")
with open(outpath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['rela', 'sab', 'count'])
    for (rela, sab), cnt in sorted(rela_sab_counter.items(), key=lambda x: -x[1]):
        writer.writerow([rela, sab, cnt])

print(f"Saved {len(rela_sab_counter):,} (RELA, SAB) rows to CSV.")