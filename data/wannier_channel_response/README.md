# Margin responses to Wannier-channel scaling

Responses χ = ∂Λ/∂ln λ of the oriented valence and conduction margins when one hopping channel of a Wannier Hamiltonian is scaled by λ = 0.75 and 1.25 (±25%), for 45 channel directions in 13 Hamiltonians (spinels, single and double perovskites); the reference k points a/b of each margin are held fixed.

- `hamiltonian_45_channel_responses.csv`: per direction the endpoint margins, χ_V and χ_C, their common and differential parts and sign relation, and the channel's group.
- `hamiltonian_operational_groups.csv`: directions, opposite-sign directions and median |χ| per channel group; `hamiltonian_direction_selection.json`: selection of the 45 directions from 168 candidates (70 without finite endpoint splitting, 53 with a change of edge identity removed).
- `hamiltonian_same_task_witnesses.csv`: Hamiltonians with one same-sign and one opposite-sign channel.
- `recompute_hamiltonian_responses.py` recomputes χ from the endpoint margins → `hamiltonian_response_recomputed.json` (45 directions, 17 opposite-sign, 8 Hamiltonians with both relations).
