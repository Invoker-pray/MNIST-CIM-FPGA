rm -rf data data_packed

python3 scripts/import_route_b_output.py ../sw/route_b_output

python3 scripts/gen_board_sample_rom.py ../sw/route_b_output data/samples/mnist_samples_route_b_output_2.hex

python3 scripts/convert_data.py \
	--data-dir data \
	--out-dir data_packed
