"""Fetch pinned EvoApprox8b C models and generate unsigned 8x8 truth tables.

EvoApprox8b contains *unsigned* multipliers. Signed INT8 inference uses a
separate sign-magnitude adapter in :mod:`src.axm.lut_multiplier`; its hardware
overhead must be included in any eventual PPA comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.request import urlopen

import numpy as np


EVOAPPROX_COMMIT = "ad6bb819bed0c641e26c882f06796efcfe406960"
RAW_BASE = f"https://raw.githubusercontent.com/ehw-fit/evoapprox8b/{EVOAPPROX_COMMIT}"
NAME_PATTERN = re.compile(r"mul8_\d{3}\Z")


def load_metadata(cache_dir: Path) -> dict[str, dict]:
    """Return multiplier descriptions from the archived official library."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "meta.json"
    if not path.exists():
        path.write_bytes(urlopen(f"{RAW_BASE}/meta.json", timeout=60).read())
    catalog = json.loads(path.read_text(encoding="utf-8"))
    items = catalog[1]["datasets"][0]["datasets"][0]["instances"]
    return {item["name"]: item for item in items if NAME_PATTERN.fullmatch(item["name"])}


def select_circuits(metadata: dict[str, dict], count: int = 50) -> list[str]:
    """Select deterministic power/error quantiles from the 505 circuits.

    The library is already a multi-metric Pareto collection. Sampling both
    power and reported mean relative error avoids taking an arbitrary first
    50 entries and gives a documented spread for the application sweep.
    """
    if count < 1 or count > len(metadata):
        raise ValueError("Circuit count must be between 1 and library size")
    selected: list[str] = []
    quota = (count + 2) // 3
    for metric in ("pwr", "mre%", "mae%"):
        ordered = sorted(metadata, key=lambda name: (float(metadata[name]["params"][metric]), name))
        for position in np.linspace(0, len(ordered) - 1, quota, dtype=int):
            name = ordered[position]
            if name not in selected:
                selected.append(name)
    power_order = sorted(metadata, key=lambda name: (float(metadata[name]["params"]["pwr"]), name))
    for name in power_order:
        if len(selected) == count:
            break
        if name not in selected:
            selected.append(name)
    return selected[:count]


def build_lut(name: str, cache_dir: Path, metadata: dict[str, dict] | None = None) -> np.ndarray:
    """Compile one official C circuit and exhaustively evaluate all 65,536 inputs.

    This function is intended for the Kaggle runtime. ``uint16`` is required:
    an unsigned 8x8 product reaches 65,025, beyond signed int16 capacity.
    """
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid EvoApprox8b circuit ID: {name!r}")
    metadata = load_metadata(cache_dir) if metadata is None else metadata
    if name not in metadata:
        raise ValueError(f"Circuit {name} is absent from the pinned official catalog")
    cache_dir.mkdir(parents=True, exist_ok=True)
    lut_path = cache_dir / f"{name}.npy"
    if lut_path.exists():
        lut = np.load(lut_path, allow_pickle=False)
        if lut.shape != (256, 256) or lut.dtype != np.uint16:
            raise ValueError(f"Invalid cached truth table: {lut_path}")
        return lut
    c_path = cache_dir / f"{name}.c"
    if not c_path.exists():
        c_path.write_bytes(urlopen(f"{RAW_BASE}/multipliers_8/source_c/{name}.c", timeout=60).read())
    wrapper = cache_dir / f"{name}_dump.c"
    wrapper.write_text(
        '#include <stdio.h>\n#include <stdint.h>\n'
        f'#include "{name}.c"\n'
        f'int main(void) {{ for (int a=0; a<256; ++a) for (int b=0; b<256; ++b) {{ '
        f'uint16_t value={name}((uint8_t)a,(uint8_t)b); '
        'if (fwrite(&value,sizeof(value),1,stdout)!=1) return 1; } return 0; }\n',
        encoding="ascii",
    )
    executable = cache_dir / f"{name}_dump"
    subprocess.run(["gcc", "-O2", "-std=c99", str(wrapper), "-o", str(executable)], check=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True)
    lut = np.frombuffer(result.stdout, dtype="<u2").reshape(256, 256).copy()
    np.save(lut_path, lut, allow_pickle=False)
    return lut


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pinned EvoApprox8b multiplier LUTs on Kaggle.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/evoapprox8b"))
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    metadata = load_metadata(args.cache_dir)
    names = select_circuits(metadata, args.count)
    (args.cache_dir / "selected_ids.json").write_text(
        json.dumps({"source_commit": EVOAPPROX_COMMIT, "ids": names}, indent=2), encoding="utf-8"
    )
    for index, name in enumerate(names, 1):
        lut = build_lut(name, args.cache_dir, metadata)
        print(f"[{index}/{len(names)}] {name}: {lut.shape}, {lut.dtype}", flush=True)


if __name__ == "__main__":
    main()
