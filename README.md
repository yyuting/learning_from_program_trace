# learning_from_program_trace
Source code for paper "Learning from Shader Program Trace"

## Reproduce model inference results

Generate script or inference commands:

    python generate_script.py --modelroot ./ --mode inference

This will generate a script.sh with inference commands. Each command is commented to explain what result it can reproduce.

## Reproduce figures / tables in paper

    python generate_result.py ./
    
This will reproduce the following result.

Fig 2: result_figs/fig2.png

All table results: result_figs/table.pdf

All qualitative results: result_figs/fig_main.pdf