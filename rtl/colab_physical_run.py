"""Colab runner: exhaustive C-vs-RTL check, then SKY130 cell implementation.

Requires the official OpenLane 2 Colab Nix profile, SKY130 Volare PDK,
Python 3.11 OpenLane venv, gcc, and Icarus Verilog. No EEG model is run here.
"""

from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

COMMIT = "ad6bb819bed0c641e26c882f06796efcfe406960"
BASE = f"https://raw.githubusercontent.com/ehw-fit/evoapprox8b/{COMMIT}/multipliers_8"
ROOT = Path("/content/eeg_axm_physical")
VENV = Path("/content/ol-venv/bin/openlane")
DESIGNS = ("exact", "mul8_348", "mul8_112", "mul8_424")
PERIOD_NS = 20.0  # 50 MHz; the exact cell did not close timing at 10 ns.


def run(args: list[str], *, cwd: Path | None = None, log: Path | None = None) -> str:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    output = proc.stdout + proc.stderr
    if log:
        log.write_text(output, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{args[0]} exited {proc.returncode}: {output[-2000:]}")
    return output


def fetch(name: str, extension: str) -> bytes:
    folder = "source_v" if extension == "v" else "source_c"
    url = f"{BASE}/{folder}/{name}.{extension}"
    data = urllib.request.urlopen(url, timeout=60).read()
    (ROOT / "sources" / f"{name}.{extension}").write_bytes(data)
    return data


def signed(value: int) -> int:
    return value if value < 128 else value - 256


def make_design(name: str) -> Path:
    folder = ROOT / "designs" / name
    folder.mkdir(parents=True, exist_ok=True)
    if name == "exact":
        core_name = "mul8_exact"
        core = "module mul8_exact(input [7:0] A, B, output [15:0] O);\nassign O = A * B;\nendmodule\n"
        multiply = lambda a, b: a * b
    else:
        core_name = name
        core = fetch(name, "v").decode("utf-8")
        # The released Verilog references two misspelled primitive names
        # without defining them. Keep that file intact and supply aliases in
        # the generated wrapper. Exhaustive C-vs-RTL simulation below decides
        # whether their inferred logic is actually correct for each circuit.
        core += (
            "\nmodule NPDKGEPDKGENNAND2X1(input A, B, output Y); "
            "assign Y = ~(A & B); endmodule\n"
            "module XNPDKGENOR2X1(input A, B, output Y); "
            "assign Y = ~(A ^ B); endmodule\n"
            "module XPDKGENOR2X1(input A, B, output Y); "
            "assign Y = A ^ B; endmodule\n"
        )
        fetch(name, "c")
        so = folder / f"{name}.so"
        run(["gcc", "-shared", "-fPIC", "-O2", str(ROOT / "sources" / f"{name}.c"), "-o", str(so)])
        lib = ctypes.CDLL(str(so))
        func = getattr(lib, name)
        func.argtypes = (ctypes.c_uint8, ctypes.c_uint8)
        func.restype = ctypes.c_uint16
        multiply = func
    wrapper = f"""
module signed_mul8_cell(input clk, input signed [7:0] a, b,
                        output reg signed [15:0] p);
  wire [7:0] a_mag = a[7] ? -a : a;
  wire [7:0] b_mag = b[7] ? -b : b;
  wire [15:0] unsigned_product;
  wire [15:0] signed_product = (a[7] ^ b[7])
      ? -unsigned_product : unsigned_product;
  {core_name} core(.A(a_mag), .B(b_mag), .O(unsigned_product));
  always @(posedge clk) p <= signed_product;
endmodule
"""
    (folder / "signed_mul8_cell.v").write_text(core + "\n" + wrapper, encoding="utf-8")
    config = {
        "DESIGN_NAME": "signed_mul8_cell",
        "VERILOG_FILES": "dir::signed_mul8_cell.v",
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": PERIOD_NS,
        "FP_CORE_UTIL": 30,
        "RUN_KLAYOUT_DRC": False,
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (folder / "gold.hex").open("w", encoding="ascii") as f:
        for ai in range(256):
            for bi in range(256):
                a, b = signed(ai), signed(bi)
                product = int(multiply(abs(a), abs(b)))
                if (a < 0) ^ (b < 0):
                    product = -product
                f.write(f"{product & 0xffff:04x}\n")
    tb = """
module tb;
  reg clk = 0;
  reg signed [7:0] a, b;
  wire signed [15:0] p;
  reg [15:0] gold [0:65535];
  integer i;
  signed_mul8_cell dut(.clk(clk), .a(a), .b(b), .p(p));
  initial begin
    $readmemh("gold.hex", gold);
    for (i=0; i<65536; i=i+1) begin
      a = i >> 8; b = i & 255;
      #1; clk=1; #1;
      if (p !== gold[i]) begin
        $display("FAIL %0d %0d got=%h expected=%h", a, b, p, gold[i]);
        $fatal(1);
      end
      clk=0;
    end
    $display("PASS 65536 signed input pairs");
    $finish;
  end
endmodule
"""
    (folder / "tb.v").write_text(tb, encoding="ascii")
    run(["iverilog", "-g2012", "-s", "tb", "-o", "simv", "signed_mul8_cell.v", "tb.v"], cwd=folder)
    result = run(["vvp", "simv"], cwd=folder, log=folder / "simulation.log")
    if "PASS 65536" not in result:
        raise RuntimeError(f"Simulation did not pass: {name}")
    print(name, "RTL-vs-C-LUT PASS", flush=True)
    return folder


def main() -> None:
    (ROOT / "sources").mkdir(parents=True, exist_ok=True)
    folders = [make_design(name) for name in DESIGNS]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PDK_ROOT"] = "/root/.volare"
    rows = []
    for name, folder in zip(DESIGNS, folders):
        with (folder / "openlane.log").open("w", encoding="utf-8") as f:
            proc = subprocess.run([str(VENV), str(folder / "config.json")],
                                  cwd=folder, env=env, text=True, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"{name} OpenLane failed; see {folder / 'openlane.log'}")
        runs = sorted((folder / "runs").glob("RUN_*/final/metrics.json"))
        if not runs:
            raise RuntimeError(f"{name} produced no final metrics")
        metrics = json.loads(runs[-1].read_text(encoding="utf-8"))
        (folder / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        area = float(metrics["design__instance__area__stdcell"])
        leakage = float(metrics["power__leakage__total"])
        internal = float(metrics["power__internal__total"])
        switching = float(metrics["power__switching__total"])
        power = leakage + internal + switching
        delay = PERIOD_NS - float(metrics["timing__setup__ws"])
        if delay <= 0:
            raise ValueError(f"Invalid setup-derived delay for {name}: {delay}")
        energy_pj = power * PERIOD_NS * 1e3  # W / clock frequency, in pJ/result
        row = dict(design=name, area_um2=area, delay_ns=delay,
                   fmax_mhz=1000 / delay, leakage_mw=leakage * 1000,
                   dynamic_mw=(internal + switching) * 1000,
                   total_mw=power * 1000, energy_pj=energy_pj,
                   edp_pj_ns=energy_pj * delay, run=str(runs[-1].parent.parent))
        rows.append(row)
        with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row)
            writer.writeheader()
            writer.writerows(rows)
        print(name, "PPA", row, flush=True)
    baseline = rows[0]
    for row in rows:
        for metric in ("area_um2", "delay_ns", "total_mw", "edp_pj_ns"):
            row[f"{metric}_reduction_pct"] = 100 * (1 - row[metric] / baseline[metric])
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    # Colab's Nix PYTHONPATH can mix Python standard libraries inside the
    # KLayout subprocess. Run the same SKY130A deck separately with it unset.
    decks = list(Path("/root/.volare/volare/sky130/versions").glob(
        "*/sky130A/libs.tech/klayout/drc/sky130A.lydrc"))
    if len(decks) != 1:
        raise RuntimeError(f"Expected one SKY130A KLayout DRC deck, found {decks}")
    klayout = shutil.which("klayout")
    if not klayout:
        raise RuntimeError("KLayout executable missing")
    drc_rows = []
    for name, folder in zip(DESIGNS, folders):
        runs = sorted((folder / "runs").glob("RUN_*/final/gds/signed_mul8_cell.gds"))
        if not runs:
            raise RuntimeError(f"No final GDS for {name}")
        report = ROOT / f"{name}_klayout_drc.lyrdb"
        log = ROOT / f"{name}_klayout_drc.log"
        result = subprocess.run(
            [klayout, "-b", "-rd", f"input={runs[-1]}", "-rd",
             f"report={report}", "-r", str(decks[0])],
            env=env, text=True, capture_output=True,
        )
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode or not report.exists():
            raise RuntimeError(f"KLayout DRC failed for {name}; see {log}")
        violations = len(ET.parse(report).findall("./items/item"))
        drc_rows.append(dict(design=name, klayout_exit=result.returncode,
                             violations=violations))
        if violations:
            raise RuntimeError(f"KLayout DRC found {violations} violations for {name}")
    with (ROOT / "klayout_drc_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=drc_rows[0])
        writer.writeheader()
        writer.writerows(drc_rows)
    print("ALL FOUR PHYSICAL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
