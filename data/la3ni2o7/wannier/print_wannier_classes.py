"""Print the class decomposition stored in wannier_<i>.json (wannier_decomposition.py output).
usage: python print_wannier_classes.py <interval> [<interval> ...]   (interval P1, P2, S1 or S2; run in the directory
holding the json files)"""
import json, sys

for interval in sys.argv[1:]:
    decomposition = json.load(open(f"wannier_{interval.lower()}.json"))
    pairs = decomposition["pairs"]
    print(f"== {interval}  (Gamma check {decomposition['gamma_check_meV']:.3f} meV)")
    print("states R:", {s: round(v["R_meV"], 1) for s, v in decomposition["states"].items()},
          " split:", {s: round(v["split_meV"]) for s, v in decomposition["states"].items()})
    names = list(next(iter(pairs.values()))["classes_meV"])
    d_b = {n: (pairs["10-00"]["classes_meV"][n] + pairs["11-01"]["classes_meV"][n]) / 2 for n in names}
    d_c = {n: (pairs["01-00"]["classes_meV"][n] + pairs["11-10"]["classes_meV"][n]) / 2 for n in names}
    cols = [p for p in pairs] + ["Δb(avg)", "Δc(avg)"]
    print("%-24s" % "class" + "".join("%9s" % c for c in cols))
    for n in names:
        print("%-24s" % n + "".join("%9.2f" % pairs[p]["classes_meV"][n] for p in pairs)
              + "%9.2f%9.2f" % (d_b[n], d_c[n]))
    print("%-24s" % "total" + "".join("%9.2f" % pairs[p]["total_meV"] for p in pairs)
          + "%9.2f%9.2f" % (sum(d_b.values()), sum(d_c.values())))
    print("%-24s" % "ends" + "".join("%9.2f" % pairs[p]["ends_meV"] for p in pairs))
