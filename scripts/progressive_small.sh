python -m srcs.train --config progressive_small --force
python -m srcs.eval --run runs/v1/progressive_s06/progressive_w128_d3_k4 --plot --data-root data/v1_small
#python -m srcs.visualization.plots eval --name baseline_w512_d2_k4_s0123