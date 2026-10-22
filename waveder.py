#!/usr/bin/env python3
"""
Parser for VASP WAVEDER files: the per-k-point, per-band-pair dipole/momentum-derivative
matrix elements <v,k|-ir|c,k> written by LOPTICS = .TRUE., consumed internally by
ALGO = BSE (bse.F/bse_driver.F) to build exciton transition dipole moments.

Modeled after pymatgen.io.vasp.outputs.Waveder
(https://pymatgen.org/pymatgen.io.vasp.html#pymatgen.io.vasp.outputs.Waveder), with two
differences:

- The matrix-element precision (VASP's `GDEF` type for CDER_BETWEEN_STATES can be compiled as
  single precision independently of the rest of the code, to save memory on this huge array)
  is auto-detected from each record's byte length by default, instead of requiring the caller
  to already know and pass `data_type=` up front. A real WAVEDER from VASP 6.6.0 was found with
  double-precision header/WPLASMON records but single-precision (complex64) CDER_BETWEEN_STATES.
- `nbands_cder` (the second band index's upper bound -- the occupied/low-band window
  CDER_BETWEEN_STATES actually covers; VASP only stores that one triangle, see below) is
  exposed directly as its own field, since BSE post-processing needs to validate transitions
  against it explicitly.

Binary format (see linear_optics.F:WRT_CDER_BETWEEN_STATES in the VASP source)::

    record 1: int32 x 4                                       -> NB_TOT, NBANDS_CDER, NKPTS, ISPIN
    record 2: real(q), 1 value                                 -> NODES_IN_DIELECTRIC_FUNCTION
    record 3: real(q), 3x3 (Fortran column-major)               -> WPLASMON
    record 4: GDEF, NB_TOT x NBANDS_CDER x NKPTS x ISPIN x 3    -> CDER_BETWEEN_STATES

Only entries with the second band index <= NBANDS_CDER are physically populated -- VASP only
computes/stores a "valence/occupied x everything" triangle (see
CDER_BETWEEN_STATES_ROTATED in linear_optics.F), not the full NB_TOT x NB_TOT matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np

DataType = Literal["complex64", "complex128", "float32", "float64", "auto"]


def _read_fortran_record(file: BinaryIO, marker_dtype: np.dtype = np.dtype("<i4")) -> bytes:
    """Read one Fortran unformatted-sequential record.

    Transparently concatenates continuation sub-records: a negative prefix/suffix marks a
    non-final chunk of the same logical record (some Fortran runtimes split very large arrays
    this way since a single record's length would otherwise overflow the marker). Mirrors
    pymatgen.io.vasp.outputs.Waveder.from_binary's internal read_data() helper.
    """
    marker_size = marker_dtype.itemsize
    data = bytearray()
    while True:
        prefix_bytes = file.read(marker_size)
        if len(prefix_bytes) < marker_size:
            raise EOFError("Unexpected EOF while reading a Fortran record marker.")
        prefix = int(np.frombuffer(prefix_bytes, dtype=marker_dtype)[0])
        chunk = file.read(abs(prefix))
        if len(chunk) != abs(prefix):
            raise EOFError(f"Unexpected EOF: expected {abs(prefix)} bytes, got {len(chunk)}.")
        data += chunk
        suffix_bytes = file.read(marker_size)
        if len(suffix_bytes) < marker_size:
            raise EOFError("Unexpected EOF while reading a Fortran record trailer.")
        suffix = int(np.frombuffer(suffix_bytes, dtype=marker_dtype)[0])
        if abs(prefix) != abs(suffix):
            raise ValueError(f"Fortran record markers do not match: prefix={prefix}, suffix={suffix}.")
        if prefix >= 0:
            break
    return bytes(data)


def _infer_real_dtype(raw: bytes, n_elements: int, label: str) -> np.dtype:
    for dtype in (np.dtype(np.float64), np.dtype(np.float32)):
        if len(raw) == n_elements * dtype.itemsize:
            return dtype
    raise ValueError(
        f"{label} record is {len(raw)} bytes, not a clean float32/float64 array of "
        f"{n_elements} elements."
    )


def _infer_complex_dtype(raw: bytes, n_elements: int) -> np.dtype:
    for dtype in (np.dtype(np.complex64), np.dtype(np.complex128)):
        if n_elements > 0 and len(raw) == n_elements * dtype.itemsize:
            return dtype
    raise ValueError(
        f"CDER_BETWEEN_STATES record is {len(raw)} bytes, not a clean complex64/complex128 "
        f"array of {n_elements} elements. Pass an explicit data_type to Waveder.from_binary "
        "if this file uses a nonstandard precision."
    )


@dataclass
class Waveder:
    """Parsed VASP WAVEDER.

    Indices of `cder`/`cder_real`/`cder_imag` are
    [band_i (0..nbands-1), band_j (0..nbands_cder-1), kpoint, spin, cartesian_direction],
    matching pymatgen.io.vasp.outputs.Waveder's convention.
    """

    cder_real: np.ndarray
    cder_imag: np.ndarray
    nbands_cder: int
    nodes_in_dielectric_function: float
    wplasmon: np.ndarray

    @classmethod
    def from_binary(cls, filename: Path | str, data_type: DataType = "auto") -> "Waveder":
        """Read a binary WAVEDER file.

        Args:
            filename: path to WAVEDER.
            data_type: precision of CDER_BETWEEN_STATES ("complex64" or "complex128"). The
                default "auto" detects it from the record's byte length -- prefer this over
                guessing, since the matrix-element precision is independent of the rest of the
                file (see module docstring) and VASP gives no in-band indication of which one
                was used.
        """
        path = Path(filename)
        with path.open("rb") as file:
            header = np.frombuffer(_read_fortran_record(file), dtype=np.int32)
            if header.size != 4:
                raise ValueError(f"WAVEDER header record has {header.size} int32 values, expected 4.")
            nb_tot, nbands_cder, nkpts, ispin = (int(x) for x in header)
            if nb_tot <= 0 or nbands_cder <= 0 or nkpts <= 0 or ispin not in (1, 2):
                raise ValueError(f"Implausible WAVEDER header {(nb_tot, nbands_cder, nkpts, ispin)}.")

            nodes_raw = _read_fortran_record(file)
            real_dtype = _infer_real_dtype(nodes_raw, n_elements=1, label="NODES_IN_DIELECTRIC_FUNCTION")
            nodes_in_dielectric_function = float(np.frombuffer(nodes_raw, dtype=real_dtype)[0])

            wplasmon_raw = _read_fortran_record(file)
            wplasmon_dtype = _infer_real_dtype(wplasmon_raw, n_elements=9, label="WPLASMON")
            wplasmon = (
                np.frombuffer(wplasmon_raw, dtype=wplasmon_dtype)
                .reshape((3, 3), order="F")
                .astype(np.float64)
            )

            cder_raw = _read_fortran_record(file)
            n_elements = nb_tot * nbands_cder * nkpts * ispin * 3
            if data_type == "auto":
                complex_dtype = _infer_complex_dtype(cder_raw, n_elements)
            else:
                complex_dtype = np.dtype(data_type)
                expected_bytes = n_elements * complex_dtype.itemsize
                if len(cder_raw) != expected_bytes:
                    raise ValueError(
                        f"CDER_BETWEEN_STATES record is {len(cder_raw)} bytes; data_type="
                        f"{data_type!r} would need {expected_bytes} bytes for {n_elements} "
                        "elements. Pass data_type='auto' (the default) to detect the precision "
                        "automatically."
                    )
            # Fortran column-major storage: first index (band_i, up to NB_TOT) varies fastest.
            cder = (
                np.frombuffer(cder_raw, dtype=complex_dtype)
                .astype(np.complex128)
                .reshape((nb_tot, nbands_cder, nkpts, ispin, 3), order="F")
            )

        return cls(
            cder_real=cder.real,
            cder_imag=cder.imag,
            nbands_cder=nbands_cder,
            nodes_in_dielectric_function=nodes_in_dielectric_function,
            wplasmon=wplasmon,
        )

    @property
    def cder(self) -> np.ndarray:
        """The complex derivative <i|-ir|j> of the orbitals with respect to k."""
        return self.cder_real + 1j * self.cder_imag

    @property
    def nbands(self) -> int:
        """Total number of bands (first band index's range)."""
        return self.cder_real.shape[0]

    @property
    def nkpoints(self) -> int:
        return self.cder_real.shape[2]

    @property
    def nspin(self) -> int:
        return self.cder_real.shape[3]

    def get_orbital_derivative_between_states(
        self,
        band_i: int,
        band_j: int,
        kpoint: int,
        spin: int,
        cart_dir: int,
    ) -> complex:
        """<band_i,kpoint|-ir|band_j,kpoint> along Cartesian direction `cart_dir`, for the
        stored (irreducible) k-point and band indices as-is (no symmetry rotation or band-index
        triangle swap -- see bsefatband.py:cder_between_states_rotated for that). All indices
        are 0-based (band_i in 0..nbands-1, band_j in 0..nbands_cder-1, kpoint in
        0..nkpoints-1, spin in 0..nspin-1, cart_dir in 0,1,2), matching pymatgen's convention.
        """
        return complex(self.cder[band_i, band_j, kpoint, spin, cart_dir])
