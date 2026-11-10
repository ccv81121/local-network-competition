# Charge of the interlayer d_z2 pairs at Gamma (Fig. 4a shading)

- **SCF:** U = 4 eV readout settings of `four_state/pbe_u4` (12x12x3 Gamma mesh, LORBIT 11) with LCHARG/LWAVE on. Run on the drive-grid structures `drive_plane_grid/{p0,p2,film}_contcar` (server `$WORK/ORB327_20261104/<s>`, CPU, VASP 6.5.1).
- **Band choice:** made with the rule of `four_state_readout.py` (`bands.json`).
  - gamma pair: bands 215/216, the top two d_z2 bands with O1 p_z < 0.03.
  - partner: bands 217/218, the lowest two with O1 p_z >= 0.08.
  - E_gamma = −24, +16 and −345 meV for p0, p2 and film. This reproduces the +40 and −321 meV of Fig. 4c.
- **Partial charge:** LPARD with IBAND 215–218, KPUSE 1, LSEPB, KPAR 1, ALGO None, from the SCF WAVECAR (`<s>/parchg`).
- **Slices:** `orbital_slices.py` builds `slices.npz` on the plane of `code/fig4.py` bilayer(). The PARCHG files (61 MB each) are kept on the server only.
