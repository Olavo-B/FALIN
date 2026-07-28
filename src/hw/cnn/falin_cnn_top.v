/*****************************************************************************/
/* File: falin_cnn_top.v                                                   */
/* Author: Olavo Alves Barros Silva                                        */
/* Contact: olavo.barros@ufv.com                                           */
/* Date: 2026-07-23                                                        */
/* Description: Top-level FalinCNN pipeline: wires conv_bn_relu_layer +     */
/* maxpool2x2 (x2), a final conv_bn_relu_layer (no pool), global_avg_pool   */
/* and classifier into the full                                            */
/*   (Conv->BN->ReLU->Requant->Pool) x2 -> Conv->BN->ReLU->Requant ->      */
/*   GlobalAvgPool -> Linear -> argmax                                     */
/* pipeline, matching FalinCNN's architecture (falin_cnn.py) exactly.       */
/* Input is the INT8 GAF image (in_flat) - designed to plug directly into  */
/* gaf_encoder.v's gaf_out_flat: export_cnn_weights.py confirmed             */
/* quant_inp's learned activation scale is exactly 1/128, matching          */
/* gaf_encoder.v's Q1.7 output convention bit-for-bit, so NO rescale is    */
/* needed at this interface.                                               */
/*                                                                          */
/* A simple top-level FSM sequences each stage's start/done handshake in   */
/* turn; each stage's out_flat is only read after its own `done`, so plain */
/* wire connections between stages are safe (no extra buffering needed).   */
/*                                                                          */
/* Multi-dataset support (added 2026-07-24): FALIN's architecture (Table I  */
/* of the paper) is IDENTICAL across all UCR datasets - only NUM_CLASSES    */
/* and the trained weight/mult/bias ROM contents differ (every dataset's    */
/* GAF image is resized to the same 96x96 footprint in software). Rather   */
/* than duplicate this file per dataset (violates "One Way"/DRY), a single  */
/* ROM_DIR string parameter is threaded down to every submodule's WEIGHT_/  */
/* MULT_/BIAS_FILE parameter via string concatenation - recompiling this    */
/* same file with a different ROM_DIR (and NUM_CLASSES) targets a          */
/* different dataset's exported weights (see export_cnn_weights.py, which  */
/* now writes to cnn_weights/<dataset>/ per dataset).                      */
/* See Task 7/8 of misc/docs/fpga_implementation_plan.md.                  */
/*****************************************************************************/

`timescale 1ns / 1ps

module falin_cnn_top #(
    parameter DATA_WIDTH   = 8,
    parameter NUM_CLASSES  = 7,
    parameter DESCRIPTOR_SIZE = 128,
    parameter GAF_SIZE     = 96,
    parameter ROM_DIR      = "cnn_weights"
) (
    input  wire                                            clk,
    input  wire                                            rst_n,
    input  wire                                            start,
    input  wire signed [1*GAF_SIZE*GAF_SIZE*DATA_WIDTH-1:0] in_flat,
    output reg                                              done,
    output reg  [$clog2(NUM_CLASSES)-1:0]                   class_out
);

    // ------------------------------------------------------------------
    // Stage 1: Conv(1->16, 96x96) -> Pool(96->48)
    // ------------------------------------------------------------------
    wire conv1_start, conv1_done;
    // Split into lo/hi packed halves (updated 2026-07-24, unpacked array
    // ports tried and rejected by both Yosys and Vivado - see
    // conv_bn_relu_layer.v's header): at 16*96*96*8 = 1,179,648 bits, one
    // packed conv1_out would be the widest inter-stage signal in the
    // design and the only one that crosses Vivado's 1,000,000-bit
    // single-variable synthesis limit ([Synth 8-4556]). conv2_out/conv3_out
    // below are split too only for port-type consistency with
    // conv_bn_relu_layer.v's now-split out_flat_lo/hi, not because they
    // individually exceed the limit (max is conv2_out at 589,824 bits).
    wire signed [8*GAF_SIZE*GAF_SIZE*DATA_WIDTH-1:0] conv1_out_lo, conv1_out_hi;

    conv_bn_relu_layer #(
        .IN_CH(1), .OUT_CH(16), .IMG_SIZE(GAF_SIZE), .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_FILE({ROM_DIR, "/layer1_weight.hex"}),
        .MULT_FILE({ROM_DIR, "/layer1_mult.hex"}),
        .BIAS_FILE({ROM_DIR, "/layer1_bias.hex"})
    ) u_conv1 (
        .clk(clk), .rst_n(rst_n), .start(conv1_start),
        .in_flat(in_flat), .done(conv1_done),
        .out_flat_lo(conv1_out_lo), .out_flat_hi(conv1_out_hi)
    );

    localparam POOL1_SIZE = GAF_SIZE / 2; // 48
    wire pool1_start, pool1_done;
    wire signed [16*POOL1_SIZE*POOL1_SIZE*DATA_WIDTH-1:0] pool1_out;

    maxpool2x2 #(
        .CH(16), .IN_SIZE(GAF_SIZE), .DATA_WIDTH(DATA_WIDTH)
    ) u_pool1 (
        .clk(clk), .rst_n(rst_n), .start(pool1_start),
        .in_flat_lo(conv1_out_lo), .in_flat_hi(conv1_out_hi),
        .done(pool1_done), .out_flat(pool1_out)
    );

    // ------------------------------------------------------------------
    // Stage 2: Conv(16->32, 48x48) -> Pool(48->24)
    // ------------------------------------------------------------------
    wire conv2_start, conv2_done;
    wire signed [16*POOL1_SIZE*POOL1_SIZE*DATA_WIDTH-1:0] conv2_out_lo, conv2_out_hi;

    conv_bn_relu_layer #(
        .IN_CH(16), .OUT_CH(32), .IMG_SIZE(POOL1_SIZE), .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_FILE({ROM_DIR, "/layer2_weight.hex"}),
        .MULT_FILE({ROM_DIR, "/layer2_mult.hex"}),
        .BIAS_FILE({ROM_DIR, "/layer2_bias.hex"})
    ) u_conv2 (
        .clk(clk), .rst_n(rst_n), .start(conv2_start),
        .in_flat(pool1_out), .done(conv2_done),
        .out_flat_lo(conv2_out_lo), .out_flat_hi(conv2_out_hi)
    );

    localparam POOL2_SIZE = POOL1_SIZE / 2; // 24
    wire pool2_start, pool2_done;
    wire signed [32*POOL2_SIZE*POOL2_SIZE*DATA_WIDTH-1:0] pool2_out;

    maxpool2x2 #(
        .CH(32), .IN_SIZE(POOL1_SIZE), .DATA_WIDTH(DATA_WIDTH)
    ) u_pool2 (
        .clk(clk), .rst_n(rst_n), .start(pool2_start),
        .in_flat_lo(conv2_out_lo), .in_flat_hi(conv2_out_hi),
        .done(pool2_done), .out_flat(pool2_out)
    );

    // ------------------------------------------------------------------
    // Stage 3: Conv(32->128, 24x24), no pool -> GlobalAvgPool -> Linear
    // ------------------------------------------------------------------
    wire conv3_start, conv3_done;
    wire signed [(DESCRIPTOR_SIZE/2)*POOL2_SIZE*POOL2_SIZE*DATA_WIDTH-1:0] conv3_out_lo, conv3_out_hi;

    conv_bn_relu_layer #(
        .IN_CH(32), .OUT_CH(DESCRIPTOR_SIZE), .IMG_SIZE(POOL2_SIZE), .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_FILE({ROM_DIR, "/layer3_weight.hex"}),
        .MULT_FILE({ROM_DIR, "/layer3_mult.hex"}),
        .BIAS_FILE({ROM_DIR, "/layer3_bias.hex"})
    ) u_conv3 (
        .clk(clk), .rst_n(rst_n), .start(conv3_start),
        .in_flat(pool2_out), .done(conv3_done),
        .out_flat_lo(conv3_out_lo), .out_flat_hi(conv3_out_hi)
    );

    wire gap_start, gap_done;
    wire signed [DESCRIPTOR_SIZE*DATA_WIDTH-1:0] gap_out;

    global_avg_pool #(
        .CH(DESCRIPTOR_SIZE), .IN_SIZE(POOL2_SIZE), .DATA_WIDTH(DATA_WIDTH),
        .MULT_FILE({ROM_DIR, "/gap_mult.hex"})
    ) u_gap (
        .clk(clk), .rst_n(rst_n), .start(gap_start),
        .in_flat_lo(conv3_out_lo), .in_flat_hi(conv3_out_hi),
        .done(gap_done), .out_flat(gap_out)
    );

    wire classifier_start, classifier_done;
    wire [$clog2(NUM_CLASSES)-1:0] classifier_class_out;

    classifier #(
        .IN_FEATURES(DESCRIPTOR_SIZE), .NUM_CLASSES(NUM_CLASSES), .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_FILE({ROM_DIR, "/classifier_weight.hex"}),
        .BIAS_FILE({ROM_DIR, "/classifier_bias.hex"})
    ) u_classifier (
        .clk(clk), .rst_n(rst_n), .start(classifier_start),
        .in_flat(gap_out), .done(classifier_done), .class_out(classifier_class_out)
    );

    // ------------------------------------------------------------------
    // Top-level FSM: sequence each stage's start/done handshake in turn.
    // ------------------------------------------------------------------
    localparam ST_IDLE       = 3'd0;
    localparam ST_CONV1      = 3'd1;
    localparam ST_POOL1      = 3'd2;
    localparam ST_CONV2      = 3'd3;
    localparam ST_POOL2      = 3'd4;
    localparam ST_CONV3      = 3'd5;
    localparam ST_GAP        = 3'd6;
    localparam ST_CLASSIFIER = 3'd7;

    reg [2:0] state;
    reg conv1_start_r, pool1_start_r, conv2_start_r, pool2_start_r;
    reg conv3_start_r, gap_start_r, classifier_start_r;

    assign conv1_start      = conv1_start_r;
    assign pool1_start      = pool1_start_r;
    assign conv2_start      = conv2_start_r;
    assign pool2_start      = pool2_start_r;
    assign conv3_start      = conv3_start_r;
    assign gap_start        = gap_start_r;
    assign classifier_start = classifier_start_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done  <= 1'b0;
            conv1_start_r <= 1'b0; pool1_start_r <= 1'b0;
            conv2_start_r <= 1'b0; pool2_start_r <= 1'b0;
            conv3_start_r <= 1'b0; gap_start_r <= 1'b0; classifier_start_r <= 1'b0;
        end else begin
            // default: all start pulses low unless explicitly asserted below
            conv1_start_r <= 1'b0; pool1_start_r <= 1'b0;
            conv2_start_r <= 1'b0; pool2_start_r <= 1'b0;
            conv3_start_r <= 1'b0; gap_start_r <= 1'b0; classifier_start_r <= 1'b0;

            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        conv1_start_r <= 1'b1;
                        state <= ST_CONV1;
                    end
                end

                ST_CONV1: if (conv1_done) begin
                    pool1_start_r <= 1'b1;
                    state <= ST_POOL1;
                end

                ST_POOL1: if (pool1_done) begin
                    conv2_start_r <= 1'b1;
                    state <= ST_CONV2;
                end

                ST_CONV2: if (conv2_done) begin
                    pool2_start_r <= 1'b1;
                    state <= ST_POOL2;
                end

                ST_POOL2: if (pool2_done) begin
                    conv3_start_r <= 1'b1;
                    state <= ST_CONV3;
                end

                ST_CONV3: if (conv3_done) begin
                    gap_start_r <= 1'b1;
                    state <= ST_GAP;
                end

                ST_GAP: if (gap_done) begin
                    classifier_start_r <= 1'b1;
                    state <= ST_CLASSIFIER;
                end

                ST_CLASSIFIER: if (classifier_done) begin
                    class_out <= classifier_class_out;
                    done      <= 1'b1;
                    state     <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
