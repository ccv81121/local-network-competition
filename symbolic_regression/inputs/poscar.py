"""Crystal structures read from POSCAR text."""
from __future__ import annotations

import base64
import gzip
import re
from dataclasses import dataclass

import numpy as np

SYMBOL = re.compile(r"^[A-Z][a-z]?$")


@dataclass(frozen=True)
class Structure:
    lattice: np.ndarray        # rows are the lattice vectors (Å)
    symbols: tuple[str, ...]   # one chemical symbol per site
    frac: np.ndarray           # fractional coordinates in [0, 1)

    @property
    def nsites(self) -> int:
        return len(self.symbols)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))


def decode(payload: str) -> str:
    """POSCAR text from a gzip-compressed, base64-encoded payload."""
    return gzip.decompress(base64.b64decode(str(payload), validate=True)).decode("utf-8")


def parse_poscar(text: str) -> Structure:
    """Parse VASP 5 POSCAR text (species line required; direct or Cartesian coordinates)."""
    lines = [line.rstrip() for line in str(text).splitlines() if line.strip()]
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float)
    if scale < 0:   # a negative scale is the cell volume
        lattice = lattice * (-scale / abs(float(np.linalg.det(lattice)))) ** (1.0 / 3.0)
    else:
        lattice = lattice * scale
    species = lines[5].split()
    counts = [int(float(x)) for x in lines[6].split()]
    if len(species) != len(counts) or not all(SYMBOL.match(s) for s in species):
        raise ValueError("POSCAR needs a species line matching the counts line")
    index = 8 if lines[7].lower().startswith("s") else 7   # optional selective-dynamics line
    mode = lines[index].strip().lower()
    n = sum(counts)
    coords = np.array([[float(x) for x in lines[index + 1 + i].split()[:3]] for i in range(n)],
                      dtype=float)
    if mode.startswith(("c", "k")):
        coords = coords @ np.linalg.inv(lattice)
    elif not mode.startswith("d"):
        raise ValueError(f"unknown POSCAR coordinate mode: {mode}")
    symbols = tuple(s for s, c in zip(species, counts) for _ in range(c))
    return Structure(lattice=lattice, symbols=symbols, frac=coords - np.floor(coords))
