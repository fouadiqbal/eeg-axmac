module mul8_exact(input [7:0] A, B, output [15:0] O);
  assign O = A * B;
endmodule


module signed_mul8_cell(input clk, input signed [7:0] a, b,
                        output reg signed [15:0] p);
  wire [7:0] a_mag = a[7] ? -a : a;
  wire [7:0] b_mag = b[7] ? -b : b;
  wire [15:0] unsigned_product;
  wire [15:0] signed_product = (a[7] ^ b[7])
      ? -unsigned_product : unsigned_product;
  mul8_exact core(.A(a_mag), .B(b_mag), .O(unsigned_product));
  always @(posedge clk) p <= signed_product;
endmodule
