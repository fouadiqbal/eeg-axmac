"""Build comparable registered signed-INT8 multiplier-cell OpenLane inputs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESIGNS = ("exact", "mul8_348", "mul8_112", "mul8_424")


def main() -> None:
    for design in DESIGNS:
        folder = ROOT / "designs" / design
        folder.mkdir(parents=True, exist_ok=True)
        core_name = "mul8_exact" if design == "exact" else design
        core = (
            "module mul8_exact(input [7:0] A, B, output [15:0] O);\n"
            "  assign O = A * B;\nendmodule\n"
            if design == "exact"
            else (ROOT / "evoapprox8b" / f"{design}.v").read_text(encoding="utf-8")
        )
        if design != "exact":
            core += (
                "\nmodule NPDKGEPDKGENNAND2X1(input A, B, output Y); "
                "assign Y = ~(A & B); endmodule\n"
                "module XNPDKGENOR2X1(input A, B, output Y); "
                "assign Y = ~(A ^ B); endmodule\n"
                "module XPDKGENOR2X1(input A, B, output Y); "
                "assign Y = A ^ B; endmodule\n"
            )
        # All four designs include the same magnitude conversion, sign restore,
        # and output register used by the software LUT path. The 8-bit core is
        # the only changed component.
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
            "CLOCK_PERIOD": 20,
            "FP_CORE_UTIL": 30,
            "RUN_KLAYOUT_DRC": False,
        }
        (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(folder)


if __name__ == "__main__":
    main()
