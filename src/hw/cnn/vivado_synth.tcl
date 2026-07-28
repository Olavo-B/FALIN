# /*****************************************************************************/
# * File: vivado_synth.tcl
# * Author: Olavo Alves Barros Silva
# * Contact: olavo.barros@ufv.com
# * Date: 2026-07-24
# * Description: Out-of-context Vivado synthesis (no block design, no
# * bitstream - see Task 8c/8d of misc/docs/fpga_implementation_plan.md) for
# * FalinCNN, targeting the real LUT/LUTRAM/FF/BRAM/DSP/BUFG numbers the
# * Yosys open-source flow (synth_xilinx) could not produce - it hangs
# * indefinitely mapping this design's wide variable-index part-selects
# * ($shiftx cells), root-caused but not worth chasing further now that
# * Vivado is available.
# *
# * Reuses the SAME per-dataset synth_top.v wrapper written by
# * export_cnn_weights.py's write_synth_wrapper_and_script (bakes in that
# * dataset's NUM_CLASSES/ROM_DIR via standard Verilog instantiation-time
# * parameter overrides - no Yosys-specific machinery involved, so it works
# * unchanged here).
# *
# * Usage: open Vivado's Tcl Console with cwd = src/hw/cnn/ (same convention
# * as every other tool invocation in this project - $readmemh paths inside
# * synth_top.v are relative to this directory), set `dataset_name` below if
# * not ElectricDevices, then:
# *   source vivado_synth.tcl
# /*****************************************************************************/

set dataset_name "ElectricDevices"
set part_name "xc7z020clg400-1"
set hw_dir "../../../results/$dataset_name/hw"

read_verilog -sv [list \
    conv_bn_relu_layer.v \
    maxpool2x2.v \
    global_avg_pool.v \
    classifier.v \
    falin_cnn_top.v \
    "$hw_dir/synth_top.v" \
]

synth_design -top falin_cnn_synth_top -part $part_name -mode out_of_context

report_utilization -file "$hw_dir/vivado_utilization_report.txt"
report_utilization -hierarchical -file "$hw_dir/vivado_utilization_hierarchical_report.txt"
report_timing_summary -file "$hw_dir/vivado_timing_summary.txt"

puts "=== Done. Reports written under $hw_dir ==="
