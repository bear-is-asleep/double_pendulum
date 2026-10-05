python -m srcs.train --config curriculum_small --force
python -m srcs.eval --run runs/curriculum_small/curriculum_* --plot
python -m srcs.visualization.plots eval --metrics runs/curriculum_small/curriculum_*/metrics.jsonl --name curriculum_small