python -m srcs.train --config baseline_small --force
python -m srcs.eval --run runs/v1/baseline_s06/baseline_w128_d3_k4 --plot --data-root data/v1_small
#python -m srcs.visualization.plots eval --name baseline_w512_d2_k4_s0123