"""Replace the structures in pressure_geometry_input_rows.csv by the tight CONTCARs (0-6 GPa).
Keeps the rows of task mp-1001019_Sb_k12 up to 6 GPa, takes the structure from ../spinel/p<p>/relax/CONTCAR (species and
counts must match) and rewrites the csv in place.
usage: python build_rows.py   (run in the geometry folder)"""
import csv

csv.field_size_limit(10 ** 9)
SOURCE_FILE = "pressure_geometry_input_rows.csv"
rows = list(csv.DictReader(open(SOURCE_FILE, encoding="utf-8-sig", newline="")))
rows = [row for row in rows if row["task"] == "mp-1001019_Sb_k12"]
print("orig rows", [row["pressure_GPa"] for row in rows])
kept_rows = []
for row in rows:
    pressure = float(row["pressure_GPa"])
    if pressure > 6.0:
        continue
    tag = "p%d" % pressure
    contcar_text = open("../spinel/%s/relax/CONTCAR" % tag).read()
    old_lines = row["poscar_text"].strip().splitlines()
    new_lines = contcar_text.strip().splitlines()
    assert (old_lines[5].split() == new_lines[5].split()
            and old_lines[6].split() == new_lines[6].split()), (old_lines[5:7], new_lines[5:7])
    row["poscar_text"] = contcar_text
    kept_rows.append(row)
writer = csv.DictWriter(open(SOURCE_FILE, "w", encoding="utf-8", newline=""), fieldnames=list(rows[0].keys()))
writer.writeheader()
writer.writerows(kept_rows)
print("tight rows", [row["pressure_GPa"] for row in kept_rows])
