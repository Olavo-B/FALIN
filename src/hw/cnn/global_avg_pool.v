/*****************************************************************************/
/* File: global_avg_pool.v                                                 */
/* Author: Olavo Alves Barros Silva                                        */
/* Contact: olavo.barros@ufv.com                                           */
/* Date: 2026-07-23                                                        */
/* Description: Global average pool over the last conv layer's INT8        */
/* activations (24x24 -> 1 per channel), folded with the requantization     */
/* into the classifier's input scale. Unlike the conv layers, GAP has no   */
/* learned scale/bias of its own, so a SINGLE scalar multiplier (not a     */
/* per-channel ROM) applied to the raw INT8 sum handles both the "/576"    */
/* average and the rescale to the classifier's input scale in one step -   */
/* see export_cnn_weights.py's header for the derivation.                  */
/* See Task 7 of misc/docs/fpga_implementation_plan.md.                    */
/*****************************************************************************/

`timescale 1ns / 1ps

module global_avg_pool #(
    parameter CH          = 128,
    parameter IN_SIZE     = 24,
    parameter DATA_WIDTH  = 8,
    parameter MULT_WIDTH  = 32,
    parameter SHIFT       = 20,      // must match export_cnn_weights.py's SHIFT
    parameter MULT_FILE   = "cnn_weights/gap_mult.hex"
) (
    input  wire                                            clk,
    input  wire                                            rst_n,
    input  wire                                            start,
    input  wire signed [(CH/2)*IN_SIZE*IN_SIZE*DATA_WIDTH-1:0] in_flat_lo,
    input  wire signed [(CH/2)*IN_SIZE*IN_SIZE*DATA_WIDTH-1:0] in_flat_hi,
    output reg                                             done,
    output reg  signed [CH*DATA_WIDTH-1:0]                 out_flat
);

    localparam CH_BITS  = (CH <= 1) ? 1 : $clog2(CH);
    localparam POS_BITS = $clog2(IN_SIZE*IN_SIZE);
    localparam HALF_CH  = CH / 2;  // CH is always even here (DESCRIPTOR_SIZE=128)

    // in_flat is split into in_flat_lo/in_flat_hi (updated 2026-07-24,
    // matches conv3's out_flat_lo/hi split - see conv_bn_relu_layer.v's
    // header for the full rationale: Vivado rejects any single packed
    // variable wider than 1,000,000 bits, and separately rejects unpacked
    // array ports outright too, so a two-way packed split is the fix).
    // ST_SUM below picks lo/hi based on which half ch_cnt falls in.

    // No separate out_buf memory + pack-loop (see conv_bn_relu_layer.v's
    // header for the full history of why: a generate-loop of per-element
    // assigns is slow for `iverilog` to elaborate at scale, and a
    // procedural pack-loop is fast for `iverilog` but is what makes
    // `yosys`'s `proc` pass take hours, since synthesis must statically
    // unroll every procedural for-loop into concrete hardware). ST_SCALE
    // below writes scale_out directly into out_flat at ch_cnt's offset.

    reg signed [MULT_WIDTH-1:0] gap_mult_rom [0:0];
    initial $readmemh(MULT_FILE, gap_mult_rom);
    wire signed [MULT_WIDTH-1:0] gap_mult = gap_mult_rom[0];

    localparam ST_IDLE   = 2'd0;
    localparam ST_SUM    = 2'd1;
    localparam ST_SCALE  = 2'd2;
    localparam ST_DONE   = 2'd3;

    reg [1:0] state;
    reg [CH_BITS-1:0]  ch_cnt;
    reg [POS_BITS-1:0] pos_cnt;
    reg signed [31:0]  acc;

    // ch_cnt is constant for the whole IN_SIZE*IN_SIZE inner sum loop of a
    // given channel, so the half-select and local channel index only need
    // to be recomputed once per channel (still fine as a continuous wire).
    wire ch_is_hi = (ch_cnt >= HALF_CH);
    wire [CH_BITS-1:0] local_ch = ch_is_hi ? (ch_cnt - HALF_CH) : ch_cnt;
    wire [31:0] sum_idx = local_ch*IN_SIZE*IN_SIZE + pos_cnt;

    localparam signed [63:0] ROUND_BIAS = 64'sd1 <<< (SHIFT - 1);
    wire signed [63:0] scaled  = acc * gap_mult;
    wire signed [63:0] rounded = scaled + ROUND_BIAS;
    wire signed [63:0] shifted = rounded >>> SHIFT;
    wire signed [DATA_WIDTH-1:0] scale_out =
        (shifted > 64'sd127) ? 8'sd127 :
        (shifted < 64'sd0)   ? 8'sd0   :
        shifted[DATA_WIDTH-1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done  <= 1'b0;
            ch_cnt  <= {CH_BITS{1'b0}};
            pos_cnt <= {POS_BITS{1'b0}};
            acc     <= 32'sd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        ch_cnt  <= {CH_BITS{1'b0}};
                        pos_cnt <= {POS_BITS{1'b0}};
                        acc     <= 32'sd0;
                        state   <= ST_SUM;
                    end
                end

                // --- Sum all IN_SIZE*IN_SIZE INT8 codes for the current channel ---
                ST_SUM: begin
                    acc <= acc + (ch_is_hi ? in_flat_hi[sum_idx*DATA_WIDTH +: DATA_WIDTH] : in_flat_lo[sum_idx*DATA_WIDTH +: DATA_WIDTH]);
                    if (pos_cnt == IN_SIZE*IN_SIZE - 1) begin
                        pos_cnt <= {POS_BITS{1'b0}};
                        state   <= ST_SCALE;
                    end else begin
                        pos_cnt <= pos_cnt + 1'b1;
                    end
                end

                ST_SCALE: begin
                    out_flat[ch_cnt*DATA_WIDTH +: DATA_WIDTH] <= scale_out;
                    acc <= 32'sd0;
                    if (ch_cnt == CH - 1) begin
                        state <= ST_DONE;
                    end else begin
                        ch_cnt <= ch_cnt + 1'b1;
                        state  <= ST_SUM;
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
