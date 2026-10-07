## Baseline observations

- Surrogate model performance is poorer near end of ellapsed time for MLP, maybe due to 
- Curriculum training requires manual tuning of the pass bar.
- Error is large right at beginning, lowest in middle, very high at end.
- Velocity equally worse at beginning and end.
- When adding more stages to curriculum, there is a rapid oscillation due to adding large fraction of stage 2 suddenly. Mitigate by gradually adding the stages
- Setting gates at each stage for curriculum is too constraining, causes large oscillations if the gates are too close, stagnates on one stage if the gates are too far apart. Doesn't allow prior stages in without much manual tuning and config.
- Overtraining of model causes validation error to maintain even when training loss is decreasing, introduce dropout to mitigate.