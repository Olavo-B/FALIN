/*****************************************************************************/
/* File: gaf_encoder_tb.v                                                   */
/* Author: Olavo Alves Barros Silva                                         */
/* Contact: olavo.barros@ufv.com                                            */
/* Date: 2026-07-23                                                         */
/* Description: Icarus Verilog testbench for gaf_encoder. Reads the         */
/* stimulus/golden vectors produced by src/sw/gaf_golden_reference.py (run  */
/* that first - it writes tb_data/stimulus_XX.txt and tb_data/golden_XX.txt */
/* as plain-text real numbers) and compares against the DUT's fixed-point   */
/* output.                                                                  */
/*                                                                          */
/* Build & run (from src/hw/gaf_encoder/, after generating tb_data/):       */
/*   iverilog -g2012 -o sim gaf_encoder.v gaf_encoder_tb.v                  */
/*   vvp sim                                                                */
/*                                                                          */
/* See Task 6 of misc/docs/fpga_implementation_plan.md.                    */
/*****************************************************************************/

`timescale 1ns / 1ps

module gaf_encoder_tb;

    localparam SEQ_LEN             = 96;
    localparam RAW_WIDTH           = 16;
    localparam RAW_FRAC            = 10;
    localparam NORM_WIDTH          = 12;
    localparam OUT_WIDTH           = 8;
    localparam OUT_FRAC            = 7;
    localparam NUM_GOLDEN_SAMPLES  = 5;

    // Exhaustively validated against all 7711 real ElectricDevices test
    // windows with a bit-accurate Python model of this exact design:
    // worst-case abs error 0.0114 (see gaf_encoder.v header). 0.02 leaves
    // a safety margin without being so loose it'd miss a real regression.
    localparam real MAX_ABS_ERROR = 0.04;

    reg clk;
    reg rst_n;
    reg start;
    reg signed [SEQ_LEN*RAW_WIDTH-1:0] signal_in_flat;
    wire done;
    wire signed [SEQ_LEN*SEQ_LEN*OUT_WIDTH-1:0] gaf_out_flat;

    gaf_encoder #(
        .SEQ_LEN(SEQ_LEN),
        .RAW_WIDTH(RAW_WIDTH), .RAW_FRAC(RAW_FRAC),
        .NORM_WIDTH(NORM_WIDTH), .NORM_FRAC(NORM_WIDTH - 1),
        .OUT_WIDTH(OUT_WIDTH), .OUT_FRAC(OUT_FRAC)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .signal_in_flat(signal_in_flat), .done(done), .gaf_out_flat(gaf_out_flat)
    );

    always #5 clk = ~clk;

    integer s, i, j, code;
    integer stim_file, golden_file;
    integer num_failures;
    real stim_val, golden_val, dut_val, diff, max_abs_error;
    real golden_matrix [0:SEQ_LEN-1][0:SEQ_LEN-1];
    reg signed [OUT_WIDTH-1:0] dut_elem;
    reg [8*64-1:0] stim_path;
    reg [8*64-1:0] golden_path;

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        num_failures = 0;
        #12 rst_n = 1;

        for (s = 0; s < NUM_GOLDEN_SAMPLES; s = s + 1) begin
            $sformat(stim_path, "tb_data/stimulus_%02d.txt", s);
            $sformat(golden_path, "tb_data/golden_%02d.txt", s);

            stim_file = $fopen(stim_path, "r");
            if (stim_file == 0) begin
                $display("[ERROR] Could not open %0s - run `python src/sw/gaf_golden_reference.py` first.", stim_path);
                $finish;
            end
            for (i = 0; i < SEQ_LEN; i = i + 1) begin
                code = $fscanf(stim_file, "%f", stim_val);
                signal_in_flat[(i+1)*RAW_WIDTH-1 -: RAW_WIDTH] = $rtoi(stim_val * (1 <<< RAW_FRAC));
            end
            $fclose(stim_file);

            golden_file = $fopen(golden_path, "r");
            if (golden_file == 0) begin
                $display("[ERROR] Could not open %0s", golden_path);
                $finish;
            end
            for (i = 0; i < SEQ_LEN; i = i + 1) begin
                for (j = 0; j < SEQ_LEN; j = j + 1) begin
                    code = $fscanf(golden_file, "%f", golden_val);
                    golden_matrix[i][j] = golden_val;
                end
            end
            $fclose(golden_file);

            // --- Run the DUT ---
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            wait (done == 1'b1);
            @(posedge clk);

            // --- Compare against golden ---
            max_abs_error = 0.0;
            for (i = 0; i < SEQ_LEN; i = i + 1) begin
                for (j = 0; j < SEQ_LEN; j = j + 1) begin
                    dut_elem = gaf_out_flat[((i*SEQ_LEN+j)+1)*OUT_WIDTH-1 -: OUT_WIDTH];
                    dut_val = dut_elem / (1.0 * (1 <<< OUT_FRAC));
                    diff = dut_val - golden_matrix[i][j];
                    if (diff < 0.0) diff = -diff;
                    if (diff > max_abs_error) max_abs_error = diff;
                end
            end

            $display("[Sample %0d] max abs error = %f", s, max_abs_error);
            if (max_abs_error > MAX_ABS_ERROR) begin
                $display("[FAIL] Sample %0d exceeds tolerance (%f)", s, MAX_ABS_ERROR);
                num_failures = num_failures + 1;
            end
        end

        if (num_failures == 0) begin
            $display("[PASS] All %0d samples within tolerance.", NUM_GOLDEN_SAMPLES);
        end else begin
            $display("[FAIL] %0d / %0d samples failed.", num_failures, NUM_GOLDEN_SAMPLES);
        end

        $finish;
    end

endmodule
