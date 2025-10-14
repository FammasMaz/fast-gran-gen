"""Parsing utilities for lunar simulant LWT particle tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class LWTRecord:
    """Single particle entry read from an ``*-LWT.dat`` file."""

    stl_path: str
    surface_area_um2: float
    volume_um3: float
    triangle_count: int
    length_um: float
    width_um: float
    thickness_um: float

    @property
    def aspect_ratio(self) -> float:
        return self.length_um / max(self.width_um, 1e-6)

    @property
    def flatness_ratio(self) -> float:
        return self.width_um / max(self.thickness_um, 1e-6)

    @property
    def equivalent_diameter_um(self) -> float:
        # Equivalent spherical diameter: d = (6V/pi)^(1/3)
        from math import pi

        return float((6.0 * self.volume_um3 / pi) ** (1.0 / 3.0))


def parse_lwt(path: Path, *, skip_invalid: bool = True) -> List[LWTRecord]:
    """Parse a lunar ``*-LWT.dat`` file into :class:`LWTRecord` instances."""

    records: List[LWTRecord] = []
    with open(path, "r", encoding="ascii", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                if skip_invalid:
                    continue
                raise ValueError(f"Row has insufficient columns: {line}")
            try:
                record = LWTRecord(
                    stl_path=parts[0],
                    surface_area_um2=float(parts[1]),
                    volume_um3=float(parts[2]),
                    triangle_count=int(float(parts[3])),
                    length_um=float(parts[4]),
                    width_um=float(parts[5]),
                    thickness_um=float(parts[6]),
                )
            except ValueError:
                if skip_invalid:
                    continue
                raise
            records.append(record)
    if not records:
        raise ValueError(f"No valid particle rows found in {path}")
    return records


def load_multiple_lwt(paths: Iterable[Path]) -> List[LWTRecord]:
    """Aggregate records from several LWT files."""

    records: List[LWTRecord] = []
    for path in paths:
        records.extend(parse_lwt(path))
    return records
