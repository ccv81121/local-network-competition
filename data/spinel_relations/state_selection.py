"""Which of the 1003 calculated spinel states the relations use, and why the others are not used.

Reads the 1003-state database, the 442 states of the relations and the 681 scored nonmetallic states, and writes
state_selection.txt: metals, the parent compounds of the 442 states, and for the other nonmetallic states their
parent compounds, the elements absent from the 442 states and the states whose gap vanishes.

usage: python data/spinel_relations/state_selection.py
"""
import collections
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
DATABASE = DATA / "number_sources" / "spinel_database" / "dataset.csv"
RELATION_STATES = DATA / "descriptor_tables" / "spinel" / "samples.csv"
SCORED = HERE / "colocation_relation" / "relation_scores_681.csv"


def read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def state_id(row):
    """sample_id of a database row: <mp_id>_k00 for the parent, <mp_id>_<substitute>_kNN along a path."""
    k = int(float(row["k"]))
    if k == 0:
        return f"{row['mp_id']}_k00"
    return f"{row['mp_id']}_{row['substitute']}_k{k:02d}"


def elements(row):
    text = row["elements"]
    for ch in "[]'\",":
        text = text.replace(ch, " ")
    return set(text.split())


def main():
    database = {state_id(r): r for r in read(DATABASE)}
    used = {r["sample_id"] for r in read(RELATION_STATES)}
    scored = {r["sample_id"] for r in read(SCORED)}
    metals = [i for i, r in database.items() if r["is_metal"] == "True"]
    nonmetals = [i for i, r in database.items() if r["is_metal"] == "False"]
    others = [i for i in nonmetals if i not in used]
    used_parents = {database[i]["mp_id"] for i in used}
    other_parents = {database[i]["mp_id"] for i in others}
    used_elements = set().union(*(elements(database[i]) for i in used))
    absent = collections.Counter()
    with_absent = 0
    for i in others:
        new = elements(database[i]) - used_elements
        absent.update(new)
        with_absent += bool(new)

    def vanishing(ids):
        return sum(database[i]["near_critical_metric_name"] == "metallic_or_zero_gap" for i in ids)

    lines = [
        f"database states: {len(database)}",
        f"metallic: {len(metals)}",
        f"nonmetallic: {len(nonmetals)}",
        f"used by the relations: {len(used)} states of {len(used_parents)} parent compounds",
        f"other nonmetallic: {len(others)} states of {len(other_parents)} parent compounds "
        f"({len(used_parents & other_parents)} parents shared with the used states; "
        f"{len([i for i in others if i not in scored])} without a scored margin)",
        f"other states containing an element absent from the used states: {with_absent} "
        f"({', '.join(f'{e} {n}' for e, n in absent.most_common())})",
        f"states with a vanishing gap: {vanishing(others)} of the other {len(others)}, {vanishing(used)} of the used {len(used)}",
    ]
    (HERE / "state_selection.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
