python -m srcs.train --config baseline_small --force
python -m srcs.eval --run runs/baseline_small/baseline_* --plot
#python -m srcs.visualization.plots eval --name baseline_w512_d2_k4_s0123