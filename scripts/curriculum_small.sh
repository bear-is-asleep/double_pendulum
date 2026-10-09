python -m srcs.train --config curriculum_small --force
python -m srcs.eval --run runs/v1/curriculum_s06/curriculum_w128_d3_k4 --plot --data-root data/v1_small