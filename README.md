# learning_from_program_trace
Source code for paper "Learning from Shader Program Trace"

## Package dependencies

The source code is developed and tested under python 3.6, TensorFlow 1.14 with CUDA 10.0. A full list of python environment can be found in environment.yml.

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

## Reference

This project uses / modifies the following repositories, our paper also cites related papers.

[1] [Fast Image Processing with Fully-Convolutional Networks] (https://github.com/CQFIO/FastImageProcessing)