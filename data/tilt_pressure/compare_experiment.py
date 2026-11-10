"""Weighted linear fit of the CaSnO3 B-O-B angles versus pressure (Zhao, Ross, Angel, Phys. Chem. Minerals 31, 299
(2004), Table 3) and the calculated 0-5 and 0-10 GPa slopes for CaSnO3 and CaTiO3."""
import numpy as np

# CaSnO3 single crystal: pressure (GPa), Sn-O-Sn angles (deg) at the O1 and O2 sites and their uncertainties
PRESSURE_GPA = np.array([0, 0.738, 1.50, 2.61, 3.94, 5.18, 5.98, 6.73])
ANGLE_O1 = np.array([147.04, 146.61, 146.49, 146.59, 146.50, 146.66, 146.58, 146.23])
ANGLE_O2 = np.array([147.87, 147.92, 147.67, 147.61, 147.43, 146.95, 147.06, 147.28])
SIGMA_O1 = np.array([.23, .22, .24, .21, .22, .19, .20, .18])
SIGMA_O2 = np.array([.13, .12, .13, .12, .12, .11, .10, .11])
for label, a, s in (("Sn-O1-Sn", ANGLE_O1, SIGMA_O1), ("Sn-O2-Sn", ANGLE_O2, SIGMA_O2),
                    ("mean(1:2)", (ANGLE_O1 + 2 * ANGLE_O2) / 3, np.sqrt(SIGMA_O1 ** 2 + 4 * SIGMA_O2 ** 2) / 3)):
    weights = 1 / s ** 2
    design = np.vstack([PRESSURE_GPA, np.ones_like(PRESSURE_GPA)]).T
    cov = np.linalg.inv(design.T @ (design * weights[:, None]))
    coefficients = cov @ design.T @ (weights * a)
    print(f"exp {label}: slope {coefficients[0]:+.3f} +- {np.sqrt(cov[0,0]):.3f} deg/GPa")
# calculated Sn-O-Sn / Ti-O-Ti angles (deg) at the O1 and O2 sites, keyed by pressure (GPa)
CALCULATED_ANGLES = {"CaSnO3": {0: (144.24, 146.17), 5: (143.74, 145.61), 10: (143.28, 145.08)},
                     "CaTiO3": {0: (154.34, 154.66), 5: (153.6, 154.16), 10: (152.96, 153.72)}}
for material, angles in CALCULATED_ANGLES.items():
    for k in (5, 10):
        print(material, f"0-{k}", [f"{(angles[k][i]-angles[0][i])/k:+.3f}" for i in (0, 1)])
