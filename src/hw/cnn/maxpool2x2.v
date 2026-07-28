/*****************************************************************************/
/* File: maxpool2x2.v                                                       */
/* Author: Olavo Alves Barros Silva                                        */
/* Contact: olavo.barros@ufv.com                                           */
/* Date: 2026-07-23                                                        */
/* Description: Generic 2x2/stride-2 max-pooling over INT8 activations.    */
/* Runs directly on INT8 codes - max is order-preserving under a positive  */
/* affine map (code = value/scale, scale > 0), so no dequantization is     */
/* needed (see export_cnn_weights.py's header for the full argument).      */
/* Separate module from conv_bn_relu_layer.v on purpose: layer3 has no     */
/* pooling after it, so folding pooling into the conv module would need a  */
/* bypass path for no benefit (Separation of Concerns).                    */
/* See Task 7 of misc/docs/fpga_implementation_plan.md.                    */
/*****************************************************************************/

`timescale 1ns / 1ps

module maxpool2x2 #(
    parameter CH          = 16,
    parameter IN_SIZE     = 96,      // must be even
    parameter DATA_WIDTH  = 8
) (
    input  wire                                                clk,
    input  wire                                                rst_n,
    input  wire                                                start,
    input  wire signed [(CH/2)*IN_SIZE*IN_SIZE*DATA_WIDTH-1:0] in_flat_lo,
    input  wire signed [(CH/2)*IN_SIZE*IN_SIZE*DATA_WIDTH-1:0] in_flat_hi,
    output reg                                                 done,
    output reg  signed [CH*(IN_SIZE/2)*(IN_SIZE/2)*DATA_WIDTH-1:0] out_flat
);

    localparam OUT_SIZE = IN_SIZE / 2;
    localparam IN_IDX_BITS  = $clog2(IN_SIZE);
    localparam OUT_IDX_BITS = (OUT_SIZE <= 1) ? 1 : $clog2(OUT_SIZE);
    localparam CH_BITS      = (CH <= 1) ? 1 : $clog2(CH);
    localparam HALF_CH      = CH / 2;  // CH is always even here (16/32)

    // in_flat is split into in_flat_lo/in_flat_hi (updated 2026-07-24 - see
    // conv_bn_relu_layer.v's header: matches its out_flat_lo/hi split,
    // needed to stay under Vivado's 1,000,000-bit single-variable limit
    // for layer1's 16*96*96*8 = 1,179,648-bit output; unpacked array ports
    // were tried first and rejected by BOTH Yosys and Vivado, for
    // different reasons - reverted). Reads below (v00..v11) pick lo/hi
    // based on which half ch_cnt falls in, then do the same variable-base
    // indexed part-select as before against that half-width bus.

    // No separate out_buf memory + pack-loop (see conv_bn_relu_layer.v's
    // header for the full history: neither a generate-loop of per-element
    // assigns nor a procedural pack-loop is right - the former is slow for
    // `iverilog` to elaborate at scale, the latter is fast for `iverilog`
    // but is what makes `yosys`'s `proc` pass take hours, since it must
    // statically unroll every procedural for-loop into concrete hardware).
    // ST_POOL below writes pool_max directly into out_flat at a
    // variable-base indexed part-select - no buffer, nothing to unroll.

    localparam ST_IDLE = 2'd0;
    localparam ST_POOL = 2'd1;
    localparam ST_DONE = 2'd2;

    reg [1:0] state;
    reg [CH_BITS-1:0]      ch_cnt;
    reg [OUT_IDX_BITS-1:0] out_row_cnt, out_col_cnt;

    wire [IN_IDX_BITS-1:0] in_row0 = {out_row_cnt, 1'b0};
    wire [IN_IDX_BITS-1:0] in_col0 = {out_col_cnt, 1'b0};

    // ch_cnt is the same for all 4 taps of a given output position, so the
    // half-select and local channel index are computed once.
    wire ch_is_hi = (ch_cnt >= HALF_CH);
    wire [CH_BITS-1:0] local_ch = ch_is_hi ? (ch_cnt - HALF_CH) : ch_cnt;

    wire [31:0] idx00 = (local_ch*IN_SIZE + in_row0)        *IN_SIZE + in_col0;
    wire [31:0] idx01 = (local_ch*IN_SIZE + in_row0)        *IN_SIZE + in_col0 + 1'b1;
    wire [31:0] idx10 = (local_ch*IN_SIZE + in_row0 + 1'b1) *IN_SIZE + in_col0;
    wire [31:0] idx11 = (local_ch*IN_SIZE + in_row0 + 1'b1) *IN_SIZE + in_col0 + 1'b1;

    wire signed [DATA_WIDTH-1:0] v00 = ch_is_hi ? in_flat_hi[idx00*DATA_WIDTH +: DATA_WIDTH] : in_flat_lo[idx00*DATA_WIDTH +: DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] v01 = ch_is_hi ? in_flat_hi[idx01*DATA_WIDTH +: DATA_WIDTH] : in_flat_lo[idx01*DATA_WIDTH +: DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] v10 = ch_is_hi ? in_flat_hi[idx10*DATA_WIDTH +: DATA_WIDTH] : in_flat_lo[idx10*DATA_WIDTH +: DATA_WIDTH];
    wire signed [DATA_WIDTH-1:0] v11 = ch_is_hi ? in_flat_hi[idx11*DATA_WIDTH +: DATA_WIDTH] : in_flat_lo[idx11*DATA_WIDTH +: DATA_WIDTH];

    wire signed [DATA_WIDTH-1:0] max01 = (v00 > v01) ? v00 : v01;
    wire signed [DATA_WIDTH-1:0] max23 = (v10 > v11) ? v10 : v11;
    wire signed [DATA_WIDTH-1:0] pool_max = (max01 > max23) ? max01 : max23;

    wire [31:0] out_elem_idx = (ch_cnt*OUT_SIZE + out_row_cnt)*OUT_SIZE + out_col_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done  <= 1'b0;
            ch_cnt <= {CH_BITS{1'b0}};
            out_row_cnt <= {OUT_IDX_BITS{1'b0}};
            out_col_cnt <= {OUT_IDX_BITS{1'b0}};
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        ch_cnt <= {CH_BITS{1'b0}};
                        out_row_cnt <= {OUT_IDX_BITS{1'b0}};
                        out_col_cnt <= {OUT_IDX_BITS{1'b0}};
                        state <= ST_POOL;
                    end
                end

                ST_POOL: begin
                    out_flat[out_elem_idx*DATA_WIDTH +: DATA_WIDTH] <= pool_max;

                    if (out_col_cnt == OUT_SIZE - 1) begin
                        out_col_cnt <= {OUT_IDX_BITS{1'b0}};
                        if (out_row_cnt == OUT_SIZE - 1) begin
                            out_row_cnt <= {OUT_IDX_BITS{1'b0}};
                            if (ch_cnt == CH - 1) begin
                                state <= ST_DONE;
                            end else begin
                                ch_cnt <= ch_cnt + 1'b1;
                            end
                        end else begin
                            out_row_cnt <= out_row_cnt + 1'b1;
                        end
                    end else begin
                        out_col_cnt <= out_col_cnt + 1'b1;
                    end
                end

                ST_DONE: begin
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
