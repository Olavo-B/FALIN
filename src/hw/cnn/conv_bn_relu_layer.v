/*****************************************************************************/
/* File: conv_bn_relu_layer.v                                              */
/* Author: Olavo Alves Barros Silva                                        */
/* Contact: olavo.barros@ufv.com                                           */
/* Date: 2026-07-23                                                        */
/* Description: Generic (parameterized) 3x3 zero-padded INT8 Conv + folded */
/* BatchNorm + ReLU + requantization layer. One instance covers all three  */
/* FalinCNN conv blocks (only IN_CH/OUT_CH/IMG_SIZE and the weight/mult/    */
/* bias ROM files change between instantiations in falin_cnn_top.v).       */
/* MaxPool is a separate module (maxpool2x2.v) - kept out of this one on   */
/* purpose (Separation of Concerns): layer3 has no pooling after it, so    */
/* baking pooling in here would need a bypass path for no real benefit.    */
/*                                                                          */
/* Math (validated in Python first - src/hw/cnn/export_cnn_weights.py -    */
/* before this file was written, same discipline as gaf_encoder.v):        */
/*   acc_int32       = sum over (in_ch,kh,kw) of x_int8 * W_int8 (3x3, pad1)*/
/*   scaled          = acc_int32 * mult_rom[out_ch] + bias_rom[out_ch]     */
/*   out_int8        = clamp(round(scaled / 2^SHIFT), 0, 127)             */
/* clamp(...,0,127) does ReLU and requantization saturation in one step   */
/* (post-ReLU only the non-negative half of the signed INT8 range is ever */
/* used - a known, accepted inefficiency of a signed activation quantizer */
/* placed after ReLU). Weight/mult/bias ROMs are produced by               */
/* export_cnn_weights.py and loaded via $readmemh, same convention as      */
/* gaf_encoder.v's sin_phi_rom.hex.                                        */
/*                                                                          */
/* Architecture: NOT a full unroll (would need OUT_CH*IMG_SIZE^2*IN_CH*9   */
/* multipliers - unusable), but not fully sequential either anymore. The   */
/* first working version did one (in_ch,kh,kw) MAC per cycle (IN_CH*9      */
/* cycles/output position) and measured out at ~33.9M cycles/sample, with  */
/* conv2 (IN_CH=16) and conv3 (IN_CH=32) alone accounting for 95% of that.  */
/* Since IN_CH is exactly the axis that varies most across layers (1/16/   */
/* 32) and kh*kw is fixed at 9 either way, the IN_CH dimension is fully    */
/* parallelized instead: for a given (kh,kw) tap, all IN_CH input channels */
/* are multiplied and summed in ONE cycle via a combinational ripple-add   */
/* chain (par_sum below - see Task 7 notes in                             */
/* misc/docs/fpga_implementation_plan.md for the before/after cycle        */
/* counts). This is the same idea as the reference paper's (Greco et al.,  */
/* misc/docs/reference/) K-parallel-multiplier-plus-tree-adder Kernel      */
/* module, just applied across IN_CH (our dominant cost axis) instead of   */
/* their K (kernel-width) axis, since our kernel is a fixed, already-small */
/* 3x3=9 taps. Cuts the inner loop from IN_CH*9 to just 9 cycles/output    */
/* position - out_ch -> row -> col -> kh -> kw remains sequential, only    */
/* in_ch stopped being a counted FSM dimension (it is now pure parallel    */
/* combinational hardware, one generate instance per channel - IN_CH is    */
/* only ever 1/16/32 here, so this is a small, cheap generate block, NOT   */
/* the large-N dynamic-index pattern that caused the earlier compile/sim   */
/* slowness - every index into par_in/par_weight/par_sum below is a        */
/* GENERATE-loop CONSTANT, never a runtime variable).                     */
/*                                                                          */
/* Ports flattened to 1D packed vectors (not unpacked SV array ports, and  */
/* NOT true `reg` arrays either) - two independent tool limits forced this: */
/* Yosys's Verilog frontend rejects unpacked array ports in the ANSI port  */
/* list, AND Vivado's synth_design rejects them too, just differently      */
/* ([Synth 8-9917] "port ... must not be declared to be an array" - tried  */
/* 2026-07-24, confirmed a dead end, reverted). Separately, Vivado also    */
/* refuses to elaborate any single PACKED variable wider than 1,000,000    */
/* bits ([Synth 8-4556]), and layer1's out_flat, if left as one packed     */
/* vector, would be 16*96*96*8 = 1,179,648 bits - over that limit (every   */
/* other inter-stage signal in the design is under it). Fix (2026-07-24):  */
/* out_flat is split into TWO packed ports, out_flat_lo/out_flat_hi, each  */
/* covering OUT_CH/2 channels (worst case 8*96*96*8 = 589,824 bits, safely */
/* under the limit) - OUT_CH is 16/32/128 across the 3 instantiations,     */
/* always even. ST_REQUANT below picks lo/hi based on which half           */
/* out_ch_cnt currently falls in. in_flat stays a single packed vector -   */
/* it never exceeds the limit in any of the 3 instantiations (max is       */
/* layer2's 294,912 bits). See maxpool2x2.v's in_flat_lo/hi and            */
/* global_avg_pool.v's in_flat_lo/hi for the matching change on the        */
/* consumer side.                                                          */
/*                                                                          */
/* See Task 7/8 of misc/docs/fpga_implementation_plan.md.                  */
/*****************************************************************************/

`timescale 1ns / 1ps

module conv_bn_relu_layer #(
    parameter IN_CH      = 1,
    parameter OUT_CH     = 16,
    parameter IMG_SIZE   = 96,       // square input/output spatial size (pad=1, stride=1 -> same size)
    parameter DATA_WIDTH = 8,        // INT8 activations (signed)
    parameter MULT_WIDTH = 32,
    parameter BIAS_WIDTH = 32,
    parameter SHIFT      = 20,       // must match export_cnn_weights.py's SHIFT
    parameter WEIGHT_FILE = "cnn_weights/layer1_weight.hex",
    parameter MULT_FILE   = "cnn_weights/layer1_mult.hex",
    parameter BIAS_FILE   = "cnn_weights/layer1_bias.hex"
) (
    input  wire                                              clk,
    input  wire                                              rst_n,
    input  wire                                              start,
    input  wire signed [IN_CH*IMG_SIZE*IMG_SIZE*DATA_WIDTH-1:0]  in_flat,
    output reg                                                done,
    output reg  signed [(OUT_CH/2)*IMG_SIZE*IMG_SIZE*DATA_WIDTH-1:0] out_flat_lo,
    output reg  signed [(OUT_CH/2)*IMG_SIZE*IMG_SIZE*DATA_WIDTH-1:0] out_flat_hi
);

    localparam IDX_BITS   = (IMG_SIZE <= 1) ? 1 : $clog2(IMG_SIZE);
    localparam OCH_BITS   = (OUT_CH   <= 1) ? 1 : $clog2(OUT_CH);
    localparam HALF_OUT_CH = OUT_CH / 2;  // OUT_CH is always even here (16/32/128)

    // ------------------------------------------------------------------
    // NOTE: the input is read directly out of in_flat below via a
    // variable-base indexed part-select (in_flat[idx*DATA_WIDTH +:
    // DATA_WIDTH]), NOT through an intermediate "wire in_buf [0:N-1]"
    // array built from a generate-loop of per-index continuous assigns.
    // That first version compiled and even elaborated fine, but was the
    // actual reason the first simulation run never produced output even
    // after the output-packing fix: a generate-loop of assigns creates N
    // *separate net objects*, and reading one of them with a RUNTIME
    // (non-constant) index forces the simulator to build an N-way
    // decoder/mux re-evaluated on every access - for layer2 that is a
    // 36864-way select, evaluated on every one of ~10M MAC cycles. Real
    // `reg` arrays (weight_rom/mult_rom/bias_rom/out_buf below) don't have
    // this problem - Icarus treats them as true indexable memory, O(1)
    // per access regardless of size. A variable part-select directly on
    // the flat vector is the same O(1) memory-like access without needing
    // a reg copy at all.
    // ------------------------------------------------------------------

    // No separate out_buf memory + pack-loop anymore (two earlier versions
    // of this file tried that: a generate-loop of one continuous assign
    // per element, and a single procedural for-loop run once - see git
    // history / misc/docs/fpga_implementation_plan.md for why each was
    // wrong: the generate version was N elaborated scopes, slow for
    // `iverilog` to elaborate at this module's scale (up to 147456
    // elements for layer1); the procedural-loop version was fast for
    // `iverilog` (which just interprets the loop at simulation runtime)
    // but is what made `yosys`'s `proc` pass take 3+ hours - synthesis
    // has to STATICALLY unroll every procedural for-loop into concrete
    // hardware at elaboration time, unlike a simulator. The actual fix:
    // don't stage results in a separate memory at all. ST_REQUANT below
    // writes each result directly into out_flat_lo or out_flat_hi (see
    // header, 2026-07-24 - split in two to stay under Vivado's 1M-bit
    // single-variable limit) at a variable-base indexed part-select, same
    // trick as before, just against a half-width bus.

    // ------------------------------------------------------------------
    // ROMs (generated by export_cnn_weights.py)
    // ------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] weight_rom [0:OUT_CH*IN_CH*9-1];
    reg signed [MULT_WIDTH-1:0] mult_rom   [0:OUT_CH-1];
    reg signed [BIAS_WIDTH-1:0] bias_rom   [0:OUT_CH-1];
    initial begin
        $readmemh(WEIGHT_FILE, weight_rom);
        $readmemh(MULT_FILE, mult_rom);
        $readmemh(BIAS_FILE, bias_rom);
    end

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    localparam ST_IDLE    = 3'd0;
    localparam ST_MAC     = 3'd1;
    localparam ST_REQUANT = 3'd2;
    localparam ST_ADVANCE = 3'd3;
    localparam ST_DONE    = 3'd4;

    reg [2:0] state;
    reg [OCH_BITS-1:0] out_ch_cnt;
    reg [IDX_BITS-1:0] row_cnt, col_cnt;
    reg [1:0] kh_cnt, kw_cnt;  // 0..2 - in_ch is no longer a counted FSM dimension, see header
    reg signed [31:0] acc;

    // --- Current MAC tap: zero-padded (pad=1) neighbor lookup ---
    wire signed [IDX_BITS+1:0] pad_row_s = $signed({2'b0, row_cnt}) + $signed({2'b0, kh_cnt}) - 2'sd1;
    wire signed [IDX_BITS+1:0] pad_col_s = $signed({2'b0, col_cnt}) + $signed({2'b0, kw_cnt}) - 2'sd1;
    wire in_bounds = (pad_row_s >= 0) && (pad_row_s < IMG_SIZE) && (pad_col_s >= 0) && (pad_col_s < IMG_SIZE);

    wire [IDX_BITS-1:0] pad_row = pad_row_s[IDX_BITS-1:0];
    wire [IDX_BITS-1:0] pad_col = pad_col_s[IDX_BITS-1:0];

    // --- All IN_CH input channels for this (kh,kw) tap, multiplied and
    // summed in ONE cycle (ripple-add chain, IN_CH terms). Every array
    // index below (gc, gc-1) is a GENERATE-loop constant, not a runtime
    // variable, so this is not the dynamic-index-into-a-wire-array
    // pattern that was slow earlier - these are just IN_CH independently
    // named point-to-point signals (IN_CH is only ever 1/16/32 here). ---
    wire signed [DATA_WIDTH-1:0] par_in     [0:IN_CH-1];
    wire signed [DATA_WIDTH-1:0] par_weight [0:IN_CH-1];
    wire signed [2*DATA_WIDTH-1:0] par_prod [0:IN_CH-1];
    wire signed [31:0] par_sum [0:IN_CH-1];

    genvar gc;
    generate
        for (gc = 0; gc < IN_CH; gc = gc + 1) begin : gen_par_mac
            wire [31:0] par_in_idx = (gc * IMG_SIZE + pad_row) * IMG_SIZE + pad_col;
            assign par_in[gc]     = in_bounds ? in_flat[par_in_idx*DATA_WIDTH +: DATA_WIDTH] : {DATA_WIDTH{1'b0}};
            assign par_weight[gc] = weight_rom[((out_ch_cnt * IN_CH + gc) * 3 + kh_cnt) * 3 + kw_cnt];
            assign par_prod[gc]   = par_in[gc] * par_weight[gc];

            if (gc == 0) begin : gen_par_sum_base
                assign par_sum[gc] = par_prod[gc];
            end else begin : gen_par_sum_chain
                assign par_sum[gc] = par_sum[gc-1] + par_prod[gc];
            end
        end
    endgenerate

    wire signed [31:0] tap_sum = par_sum[IN_CH-1];

    // --- Which half-bus this output channel belongs to, and its index
    // within that half (used by ST_REQUANT to pick/address out_flat_lo vs
    // out_flat_hi - see header, 2026-07-24) ---
    wire out_ch_is_hi = (out_ch_cnt >= HALF_OUT_CH);
    wire [OCH_BITS-1:0] local_out_ch = out_ch_is_hi ? (out_ch_cnt - HALF_OUT_CH) : out_ch_cnt;
    wire [31:0] out_elem_idx = (local_out_ch * IMG_SIZE + row_cnt) * IMG_SIZE + col_cnt;

    // --- Requantization datapath (combinational, latched in ST_REQUANT) ---
    wire signed [MULT_WIDTH-1:0] cur_mult = mult_rom[out_ch_cnt];
    wire signed [BIAS_WIDTH-1:0] cur_bias = bias_rom[out_ch_cnt];
    wire signed [63:0] scaled  = acc * cur_mult + cur_bias;
    localparam signed [63:0] ROUND_BIAS = 64'sd1 <<< (SHIFT - 1);
    wire signed [63:0] rounded = scaled + ROUND_BIAS;
    wire signed [63:0] shifted = rounded >>> SHIFT;
    wire signed [DATA_WIDTH-1:0] requant_out =
        (shifted > 64'sd127) ? 8'sd127 :
        (shifted < 64'sd0)   ? 8'sd0   :
        shifted[DATA_WIDTH-1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done  <= 1'b0;
            out_ch_cnt <= {OCH_BITS{1'b0}};
            row_cnt    <= {IDX_BITS{1'b0}};
            col_cnt    <= {IDX_BITS{1'b0}};
            kh_cnt     <= 2'd0;
            kw_cnt     <= 2'd0;
            acc        <= 32'sd0;
        end else begin
            case (state)

                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        out_ch_cnt <= {OCH_BITS{1'b0}};
                        row_cnt    <= {IDX_BITS{1'b0}};
                        col_cnt    <= {IDX_BITS{1'b0}};
                        kh_cnt     <= 2'd0;
                        kw_cnt     <= 2'd0;
                        acc        <= 32'sd0;
                        state      <= ST_MAC;
                    end
                end

                // --- Accumulate one (kh,kw) tap per cycle - all IN_CH
                // channels for that tap are summed in parallel (tap_sum) ---
                ST_MAC: begin
                    acc <= acc + tap_sum;

                    if (kw_cnt == 2'd2) begin
                        kw_cnt <= 2'd0;
                        if (kh_cnt == 2'd2) begin
                            kh_cnt <= 2'd0;
                            state  <= ST_REQUANT;
                        end else begin
                            kh_cnt <= kh_cnt + 1'b1;
                        end
                    end else begin
                        kw_cnt <= kw_cnt + 1'b1;
                    end
                end

                // --- acc now holds the full 3x3xIN_CH sum; requantize and
                // write directly into out_flat at this position's offset ---
                ST_REQUANT: begin
                    if (out_ch_is_hi)
                        out_flat_hi[out_elem_idx*DATA_WIDTH +: DATA_WIDTH] <= requant_out;
                    else
                        out_flat_lo[out_elem_idx*DATA_WIDTH +: DATA_WIDTH] <= requant_out;
                    acc   <= 32'sd0;
                    state <= ST_ADVANCE;
                end

                // --- Move to the next output position (col -> row -> out_ch) ---
                ST_ADVANCE: begin
                    if (col_cnt == IMG_SIZE - 1) begin
                        col_cnt <= {IDX_BITS{1'b0}};
                        if (row_cnt == IMG_SIZE - 1) begin
                            row_cnt <= {IDX_BITS{1'b0}};
                            if (out_ch_cnt == OUT_CH - 1) begin
                                state <= ST_DONE;
                            end else begin
                                out_ch_cnt <= out_ch_cnt + 1'b1;
                                state      <= ST_MAC;
                            end
                        end else begin
                            row_cnt <= row_cnt + 1'b1;
                            state   <= ST_MAC;
                        end
                    end else begin
                        col_cnt <= col_cnt + 1'b1;
                        state   <= ST_MAC;
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
