# learning_from_program_trace
Source code for Eurographics 2022 paper "Learning from Shader Program Trace"

## Package dependencies

The source code is developed and tested under python 3.6, TensorFlow 1.14 with CUDA 10.0. A full list of python environment can be found in environment.yml.

## Trained Models and Datasets

Trained models and testing datasets (along with a copy of the source code) can be downloaded from the following Google drive link (4GB):

[https://drive.google.com/file/d/1MHqX_ZCxxEivcygFkWObNKh_ZV872eQu/view?usp=sharing](https://drive.google.com/file/d/1MHqX_ZCxxEivcygFkWObNKh_ZV872eQu/view?usp=sharing)

## Reproduce model inference results

Generate script for inference commands:

    python generate_script.py --modelroot ./ --mode inference

This will generate a script.sh with inference commands. Each command is commented to explain what result it can reproduce.

NOTE: the script only runs properly if trained model and datasets are downloaded.

## Reproduce figures / tables in paper

    python generate_result.py ./
    
This will reproduce the following result.

Fig 1 right: result_figs/fig1.png

All table results: result_figs/table.pdf

All qualitative results: result_figs/fig_main.pdf

NOTE: this command only runs properly if trained model and datasets are downloaded.

## Re-sample and Re-generate datasets

Generate script for re-sampling and re-generating training / testing / validation datasets

    python generate_script.py --modelroot ./ --mode prepare_dataset

## Re-train model

Generate script for re-training model. Note due to anonymous Google drive's storage limit, we did not include training dataset. Actually running the command will fail unless training dataset is re-generated. We will release the entire dataset upon publication.

    python generate_script.py --modelroot ./ --mode train
    
NOTE: the script only runs properly if training datasets are available (which will be released upon publication). Training datasets can also be re-genrenerated using the commands above.
    
## Validate model

Generate script to run validation on trained intermediate models. Note due to anonymous Google drive's storage limit, we did not include validation dataset. Actually running the command will fail unless validation dataset is re-generated. We will release the entire dataset upon publication.

    python generate_script.py --modelroot ./ --mode validation
    
NOTE: the script only runs properly if validation datasets are available (which will be released upon publication). Validation datasets can also be re-genrenerated using the commands above.
    
## Profile inference runtime

Generate script that accurately estimates inference runtime.

    python generate_script.py --modelroot ./ --mode accurate_timing
    
NOTE: this command only runs properly if trained model and datasets are downloaded.

## Code Reference

This project uses / modifies the following repositories, our manuscript also cites related papers.

[1] [Fast Image Processing with Fully-Convolutional Networks](https://github.com/CQFIO/FastImageProcessing) (MIT License)

[2] [lpips-tensorflow](https://github.com/alexlee-gk/lpips-tensorflow) (BSD-2-Clause License)

[3] [MLNet-Pytorch](https://github.com/immortal3/MLNet-Pytorch) (Apache-2.0 License)

[4] [simplexnoise](https://github.com/pinae/simplexnoise) (GPL-3.0 License)

## Citation and Bibtex

Yuting Yang, Connelly Barnes and Adam Finkelstein.
"Learning from Shader Program Traces."
Eurographics, to appear, April 2022 , April, 2022.

@inproceedings{Yang:2022:LFS,
   author = "Yuting Yang and Connelly Barnes and Adam Finkelstein",
   title = "Learning from Shader Program Traces",
   booktitle = "Eurographics, to appear",
   year = "2022",
   month = apr
}
