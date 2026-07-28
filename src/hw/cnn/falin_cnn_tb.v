/*****************************************************************************/
/* File: falin_cnn_tb.v                                                    */
/* Author: Olavo Alves Barros Silva                                        */
/* Contact: olavo.barros@ufv.com                                           */
/* Date: 2026-07-23                                                        */
/* Description: Icarus Verilog testbench for falin_cnn_top. Reads the      */
/* INT8 GAF stimulus / predicted-class golden vectors produced by          */
/* src/hw/cnn/export_cnn_weights.py (run that first - it writes             */
/* tb_data/cnn_stimulus_XX.txt, 96 rows of 96 space-separated decimal INT8  */
/* codes, and tb_data/cnn_golden_XX.txt, the expected argmax class index)   */
/* and compares against the DUT's class_out.                               */
/*                                                                          */
/* NOTE ON RUNTIME: falin_cnn_top is a sequential, one-MAC-per-cycle        */
/* accelerator (see conv_bn_relu_layer.v's header) - a single forward pass  */
/* is on the order of tens of millions of clock cycles (dominated by conv2 */
/* /conv3's IN_CH*OUT_CH*IMG_SIZE^2*9 MAC count). Expect this testbench to  */
/* take real wall-clock minutes to run per sample, not seconds - this is    */
/* the expected area/speed trade-off of the simple architecture, not a bug.*/
/*                                                                          */
/* Build & run (from src/hw/cnn/, after generating tb_data/ and             */
/* cnn_weights/ via export_cnn_weights.py):                                 */
/*   iverilog -g2012 -o sim conv_bn_relu_layer.v maxpool2x2.v \             */
/*     global_avg_pool.v classifier.v falin_cnn_top.v falin_cnn_tb.v        */
/*   vvp sim                                                                */
/*                                                                          */
/* Multi-dataset support (added 2026-07-24): NUM_CLASSES/ROM_DIR/TB_DIR are */
/* now `define macros instead of hardcoded localparams, so this SAME file   */
/* recompiles for a different dataset via `-D` overrides, without           */
/* duplicating the testbench per dataset. export_cnn_weights.py writes each */
/* dataset's ROMs/testbench vectors to results/<dataset>/hw/cnn_weights/    */
/* and results/<dataset>/hw/tb_data/, and PRINTS the exact iverilog command  */
/* (with correct relative paths from src/hw/cnn/) after each run - copy     */
/* that instead of hand-writing paths here. Defaults below (7 classes,      */
/* cnn_weights/, tb_data/) match the original ElectricDevices invocation    */
/* (relative to src/hw/cnn/, back when ROMs still lived there) so it still  */
/* works unchanged with no -D flags for quick manual testing.               */
/* See Task 7/8 of misc/docs/fpga_implementation_plan.md.                  */
/*****************************************************************************/

`timescale 1ns / 1ps

`ifndef NUM_CLASSES
`define NUM_CLASSES 7
`endif
`ifndef ROM_DIR
`define ROM_DIR "cnn_weights"
`endif
`ifndef TB_DIR
`define TB_DIR "tb_data"
`endif

module falin_cnn_tb;

    localparam GAF_SIZE        = 96;
    localparam DATA_WIDTH      = 8;
    localparam NUM_CLASSES     = `NUM_CLASSES;
    localparam DESCRIPTOR_SIZE = 128;
    localparam NUM_GOLDEN_SAMPLES = 5;

    reg clk;
    reg rst_n;
    reg start;
    reg signed [1*GAF_SIZE*GAF_SIZE*DATA_WIDTH-1:0] in_flat;
    wire done;
    wire [$clog2(NUM_CLASSES)-1:0] class_out;

    falin_cnn_top #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_CLASSES(NUM_CLASSES),
        .DESCRIPTOR_SIZE(DESCRIPTOR_SIZE), .GAF_SIZE(GAF_SIZE),
        .ROM_DIR(`ROM_DIR)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_flat(in_flat), .done(done), .class_out(class_out)
    );

    always #5 clk = ~clk;

    // --- Progress heartbeat: prints whenever falin_cnn_top's own top-level
    // FSM moves to the next stage (conv1->pool1->conv2->...), plus a cycle
    // count, so a long run is visibly making progress instead of looking
    // stuck. Accesses dut.state via hierarchical reference (fine in
    // simulation, not something that needs to synthesize). ---
    reg [63:0] cycle_count;
    reg [2:0]  prev_top_state;
    always @(posedge clk) cycle_count <= cycle_count + 1'b1;
    always @(posedge clk) begin
        if (dut.state !== prev_top_state) begin
            $display("[t=%0t, cycle=%0d] falin_cnn_top state -> %0d", $time, cycle_count, dut.state);
            prev_top_state <= dut.state;
        end
    end

    integer s, i, j, code;
    integer stim_file, golden_file;
    integer num_failures;
    integer stim_val, golden_val;
    reg [8*64-1:0] stim_path;
    reg [8*64-1:0] golden_path;

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        num_failures = 0;
        #12 rst_n = 1;

        for (s = 0; s < NUM_GOLDEN_SAMPLES; s = s + 1) begin
            $sformat(stim_path, "%0s/cnn_stimulus_%02d.txt", `TB_DIR, s);
            $sformat(golden_path, "%0s/cnn_golden_%02d.txt", `TB_DIR, s);

            stim_file = $fopen(stim_path, "r");
            if (stim_file == 0) begin
                $display("[ERROR] Could not open %0s - run `python src/hw/cnn/export_cnn_weights.py` first.", stim_path);
                $finish;
            end
            for (i = 0; i < GAF_SIZE; i = i + 1) begin
                for (j = 0; j < GAF_SIZE; j = j + 1) begin
                    code = $fscanf(stim_file, "%d", stim_val);
                    in_flat[((i*GAF_SIZE+j)+1)*DATA_WIDTH-1 -: DATA_WIDTH] = stim_val[DATA_WIDTH-1:0];
                end
            end
            $fclose(stim_file);

            golden_file = $fopen(golden_path, "r");
            if (golden_file == 0) begin
                $display("[ERROR] Could not open %0s", golden_path);
                $finish;
            end
            code = $fscanf(golden_file, "%d", golden_val);
            $fclose(golden_file);

            // --- Run the DUT ---
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            wait (done == 1'b1);
            @(posedge clk);

            $display("[Sample %0d] predicted=%0d golden=%0d", s, class_out, golden_val);
            if (class_out !== golden_val[$clog2(NUM_CLASSES)-1:0]) begin
                $display("[FAIL] Sample %0d: predicted class does not match golden.", s);
                num_failures = num_failures + 1;
            end
        end

        if (num_failures == 0) begin
            $display("[PASS] All %0d samples match the golden argmax class.", NUM_GOLDEN_SAMPLES);
        end else begin
            $display("[FAIL] %0d / %0d samples failed.", num_failures, NUM_GOLDEN_SAMPLES);
        end

        $finish;
    end

endmodule
