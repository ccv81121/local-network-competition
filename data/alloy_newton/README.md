# Second Newton step and enrichment factor for the alloy intervals

Continuation of `alloy_leverage` for the intervals S₀₅→S₀₆ and S₁₁→S₁₂. Part A takes one chord (Newton) step with the same Hessian: r₁ = F<sub>aff</sub> + u<sub>pred</sub>, f₁ = DFT force at r₁, u₂ = H⁺Qf₁, r₂ = r₁ + u₂; readout and force statics at r₂. Part B computes, without new calculations, the bond-preserving share of the drive, of u<sub>pred</sub> and of u<sub>act</sub> and the enrichment factors ε = share(u)/share(drive), with the definitions of `bending_enrichment/enrichment_all.py`.

- `newton_result.json` (part A, `newton_chord_step.py prepare|stage2|eval`) and `eps_result.json` (part B, `enrichment_factor.py`); summary tables in `result.md`.
