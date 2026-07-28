/*****************************************************************************/
/* File: classifier.v                                                      */
/* Author: Olavo Alves Barros Silva                                        */
/* Contact: olavo.barros@ufv.com                                           */
/* Date: 2026-07-23                                                        */
/* Description: Final Linear(IN_FEATURES,NUM_CLASSES) + argmax. Uses       */
/* Brevitas's Int32Bias convention: bias scale is DEFINED as                */
/* input_scale*weight_scale, so bias_int32 (from export_cnn_weights.py) is  */
/* already in the same integer domain as the INT8 dot-product accumulator  */
/* - just add it directly, no per-class rescale. Every class shares the    */
/* same combined scale, so argmax over raw INT32 logits equals argmax over */
/* the true dequantized logits - the classifier needs nothing beyond a     */
/* running-max comparator, no final requantization stage at all.           */
/* See Task 7 of misc/docs/fpga_implementation_plan.md.                    */
/*****************************************************************************/

`timescale 1ns / 1ps

module classifier #(
    parameter IN_FEATURES = 128,
    parameter NUM_CLASSES = 7,
    parameter DATA_WIDTH  = 8,
    parameter BIAS_WIDTH  = 32,
    parameter WEIGHT_FILE = "cnn_weights/classifier_weight.hex",
    parameter BIAS_FILE   = "cnn_weights/classifier_bias.hex"
) (
    input  wire                                        clk,
    input  wire                                        rst_n,
    input  wire                                        start,
    input  wire signed [IN_FEATURES*DATA_WIDTH-1:0]    in_flat,
    output reg                                         done,
    output reg  [$clog2(NUM_CLASSES)-1:0]              class_out
);

    localparam CLASS_BITS = (NUM_CLASSES <= 1) ? 1 : $clog2(NUM_CLASSES);
    localparam FEAT_BITS  = (IN_FEATURES <= 1) ? 1 : $clog2(IN_FEATURES);

    wire signed [DATA_WIDTH-1:0] in_buf [0:IN_FEATURES-1];
    genvar gi;
    generate
        for (gi = 0; gi < IN_FEATURES; gi = gi + 1) begin : gen_unpack_in
            assign in_buf[gi] = in_flat[(gi+1)*DATA_WIDTH-1 -: DATA_WIDTH];
        end
    endgenerate

    reg signed [DATA_WIDTH-1:0] weight_rom [0:NUM_CLASSES*IN_FEATURES-1];
    reg signed [BIAS_WIDTH-1:0] bias_rom   [0:NUM_CLASSES-1];
    initial begin
        $readmemh(WEIGHT_FILE, weight_rom);
        $readmemh(BIAS_FILE, bias_rom);
    end

    localparam ST_IDLE = 2'd0;
    localparam ST_MAC   = 2'd1;
    localparam ST_EVAL   = 2'd2;
    localparam ST_DONE   = 2'd3;

    reg [1:0] state;
    reg [CLASS_BITS-1:0] class_cnt;
    reg [FEAT_BITS-1:0]  feat_cnt;
    reg signed [31:0]    acc;
    reg signed [31:0]    best_logit;
    reg [CLASS_BITS-1:0] best_class;

    wire signed [DATA_WIDTH-1:0] cur_in_val = in_buf[feat_cnt];
    wire signed [DATA_WIDTH-1:0] cur_weight = weight_rom[class_cnt*IN_FEATURES + feat_cnt];
    wire signed [31:0] mac_product = cur_in_val * cur_weight;
    wire signed [31:0] final_logit = acc + bias_rom[class_cnt];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done  <= 1'b0;
            class_cnt  <= {CLASS_BITS{1'b0}};
            feat_cnt   <= {FEAT_BITS{1'b0}};
            acc        <= 32'sd0;
            best_logit <= 32'sd0;
            best_class <= {CLASS_BITS{1'b0}};
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        class_cnt  <= {CLASS_BITS{1'b0}};
                        feat_cnt   <= {FEAT_BITS{1'b0}};
                        acc        <= 32'sd0;
                        best_logit <= 32'sd0;
                        best_class <= {CLASS_BITS{1'b0}};
                        state      <= ST_MAC;
                    end
                end

                // --- Dot product accumulation, one feature per cycle ---
                ST_MAC: begin
                    acc <= acc + mac_product;
                    if (feat_cnt == IN_FEATURES - 1) begin
                        feat_cnt <= {FEAT_BITS{1'b0}};
                        state    <= ST_EVAL;
                    end else begin
                        feat_cnt <= feat_cnt + 1'b1;
                    end
                end

                // --- Add bias, update running max/argmax ---
                ST_EVAL: begin
                    if (class_cnt == {CLASS_BITS{1'b0}} || final_logit > best_logit) begin
                        best_logit <= final_logit;
                        best_class <= class_cnt;
                    end
                    acc <= 32'sd0;

                    if (class_cnt == NUM_CLASSES - 1) begin
                        state <= ST_DONE;
                    end else begin
                        class_cnt <= class_cnt + 1'b1;
                        state     <= ST_MAC;
                    end
                end

                ST_DONE: begin
                    class_out <= best_class;
                    done      <= 1'b1;
                    state     <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
