# Second Newton step and enrichment factor ε for the alloy intervals

## A. Second force-balance step (same H, one chord step)

| Interval | Residual force \|Qf₀\| → \|Qf₁\| → \|Qf₂\| (eV/Å) | Distance to S_b, r₁ → r₂ (Å) | cos(u, u_act), one step → two steps | Amplitude ratio, one step → two steps |
|---|---|---|---|---|
| S₀₅→S₀₆ | 2.009 → 0.296 → 0.087 | 0.085 → 0.063 | 0.980 → 0.987 | 0.895 → 0.964 |
| S₁₁→S₁₂ | 2.355 → 0.379 → 0.122 | 0.140 → 0.126 | 0.942 → 0.951 | 0.870 → 0.971 |

Switching-edge margin (meV, change from F_aff):

| Interval | One step r₁ | Two steps r₂ | Actual S_b | Error r₁ → r₂ | Anharmonic part Λ(r₂) − Λ(r₁) |
|---|---|---|---|---|---|
| S₀₅→S₀₆, Λ_V | 11.01 | 11.50 | 11.01 | +0.01 → +0.50 | +0.49 |
| S₁₁→S₁₂, Λ_C | 28.28 | 31.31 | 33.25 | −4.97 → −1.94 | +3.03 |

Non-switching edges: the error of Λ_C for S₀₅→S₀₆ changes from 0.83 to 0.43 meV, that of Λ_V for S₁₁→S₁₂ from −6.55 to +1.37 meV.
The second step lowers the DFT energy by 6.8 and 10.4 meV.

## B. Enrichment factor ε (no new calculations; definitions as in `bending_enrichment/enrichment_all.py`)

| Interval | Bond-preserving share of the drive | Predicted | Actual | ε_pred | ε_act | Class |
|---|---|---|---|---|---|---|
| S₀₅→S₀₆ | 2.04% | 34.8% | 26.6% | 17.1 | 13.1 | E1 |
| S₁₁→S₁₂ | 0.64% | 20.3% | 10.9% | 31.9 | 17.2 | E1 |

Class E1: ε_act ≥ 9; E2: 3 ≤ ε_act < 9; E3: ε_act < 3.

Data: `newton_result.json` (`newton_chord_step.py`), `eps_result.json` (`enrichment_factor.py`).
