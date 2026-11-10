# DISPERSION_20261104 against prereg_dispersion.md (analysis only; script `analyse_dispersion_v1.py`, numbers `dispersion_result.json`)

Host H rebuilt from 24 displacements (+-0.01 A, amplitude verified from the POSCARs) by the 32 operations. Checks: orbit-fill spread (same H block reached by different operations) 1.2e-4 of max |H|; asymmetry before symmetrising 2.1e-4 (Frobenius, relative); acoustic-sum-rule row sums 1.7e-4 of max |H| (Frobenius ratio 8e-5). Translations removed by the Q projector as in checks.py. Reference forces are 7e-4 eV/A at most.
Check of the shares on the stored data: s_f = 0.0891 %, s_u(actual) = 24.80 % (both reproduce geometry.json).

| Test | Number(s) | Verdict |
|---|---|---|
| T1 | cos(u_pred, u_actual) = 0.993 over all 80 atoms (threshold 0.90); |u_pred|/|u_act| = 1.17 | PASS |
| T2 | s_u(u_pred) = 25.3 % (window 17.4-32.2 %); actual 24.8 %; s_f = 0.089 %, so enrichment 284 against the measured 278 | PASS |
| T3 | actual u: exp xi = 1.71 A, R2 = 0.995 (power law n = 1.89, R2 = 0.987). Kernel (Fourier-interpolated H(q), 6x10x6 q mesh, same 0-8.6 A range): exp xi = 6.29 A, R2 = 0.797 | INCONCLUSIVE |
| T4 | dominant q of the bond-preserving weight of u_pred: X20 (1/2,0,0), 51.2 % of s_u; then Z20 25.8 %, U20 19.3 %, Gamma20 3.6 % | PASS (with my operationalisation, see below) |

**T1.** The host filter with a pure source reproduces the relaxed displacement almost exactly in direction (0.993) and to 17 % in magnitude. Closed-loop claim stands for the substitution drive; the substitution entry is not to be withdrawn.

**T2.** The 20-atom Gamma-only value (10.3 %) is replaced by 25.3 % once the supercell H is used, so the factor-2.4 gap is a sampling artefact of Gamma-only H, as registered. Nine eigenvalues of the host H are negative (-1.36 to -0.01 eV/A^2; Ti-dominated, Gamma20 and U20 sectors, Pnma at the parent lattice) but f has no weight in them (fraction of f^2 below 1e-29), so u_pred is unchanged when they are dropped or inverted in absolute value (cos 0.993, s_u 25.34 % in both).

**T3.** Inconclusive by both registered rules, applied to the kernel: xi (6.29 A) exceeds 0.5 x the shortest edge (3.84 A) and R2 < 0.8. The kernel is a poor object here: 309 of 360 q points have a negative eigenvalue of the interpolated H(q) (minimum -1.33 eV/A^2, Ti off-centering branch), so the signed inverse has near-singular denominators, and the kernel plateaus near 0.2 A beyond 5 A rather than decaying. The actual u decays fast (xi 1.71 A, below the limit, R2 0.995) but only 7 shells up to 8.6 A are available, and with images of the source in the periodic cell. Sensitivity, not used for the verdict: inverting H(q) with |w| gives xi = 3.64 A, R2 = 0.94 over the same range, ratio 2.1 to the actual, which would be a fail if the kernel were accepted as resolved; over 0-20 A it has R2 = 0.79 and xi = 6.4 A. A larger cell is required for a decay-length statement.

**T4.** The registered Gamma-X-M-R labels are aristotype labels, all folded onto Gamma of the 20-atom cell; the 2x1x2 supercell adds the three zone-boundary points of the 20-atom cell (X20 = (1/2,0,0), Z20 = (0,0,1/2), U20 = (1/2,0,1/2); qy = 0 only). Sector weights (network-centred vectors, pi = weight, s = bond-preserving share inside the sector):

| sector | pi_f | pi_u | s_u | bond-preserving weight / s_u total | bond-preserving part in rotation span |
|---|---|---|---|---|---|
| Gamma20 | 0.251 | 0.164 | 0.056 | 3.6 % | 0.97 |
| X20 | 0.248 | 0.392 | 0.331 | 51.2 % | 0.52 |
| Z20 | 0.248 | 0.214 | 0.305 | 25.8 % | 0.26 |
| U20 | 0.253 | 0.229 | 0.214 | 19.3 % | 0.92 |

The actual u gives the same picture (X20 47.5 %, Z20 27.4 %, U20 20.9 %, Gamma20 4.2 %). Sum rule: sum of pi s = 25.34 % = s_u(u_pred) exactly; projectors idempotent, complete, commuting with H (8.9e-16) and with the bond-kernel projector (6e-16). The bond-preserving response is carried at the zone boundary and not at Gamma20, as predicted. The rotation-span overlap (single-octahedron rotations about B, as closed_loop.py; random-direction baseline 0.31) is 0.52 at the dominant X20, only marginally above the 0.5 I used, and 0.92 at U20; Z20 (0.26) is below baseline. So the statement "carried by zone-boundary sectors, not Gamma" is supported; "one pattern of rigid octahedral rotations" is supported only weakly at X20 (the dominant sector is about half rotation-like, half other bond-preserving motion) and the wording should not claim a single rigid-rotation pattern without that qualification.
Inside Gamma20 the response splits as Ag-type (eps = +++) pi 0.072 and (--+) pi 0.059 with s = 0.137; zone-boundary weight is 84 %.
