# EKRM: Expert K-Means Reconstruction Method

This is the code for method Expert K-Means Reconstruction Method (EKRM).

This code is designed for mesostructure reconstruction, providing a high-resolution image of granite as an example.

The code implements the algorithm from the paper titled "Expert K-Means Reconstruction Method: A Novel Image Processing Approach for Mesostructure Reconstruction."

This paper is currently under review, and further updates on its publication status will be available soon.

## Requirements

The code requires the following environment to run:

* Python===3.10
* Cuda==11.2
* cupy-cuda11x
* numpy==1.23.2
* scipy==1.9.1
* scikit-learn==1.1.2
* Pillow==9.2.0
* matplotlib==3.5.3
* opencv-python==4.7.0.72
* pandas==2.0.0
* tqdm==4.65.0
* numba==0.57.1

## Usage

A [demo directory](./demo) with the source image and result is provided. The files within the directory are:

* [Source optical image](./demo/befast-granite.jpg)
* [Reconstruction result](./demo/befast-granite-result.png)
* [Script 1: Config](./demo/s1_config.py)
* [Script 2: Trainer](./demo/s2_trainer.py)
* [Script 3: Predictor](./demo/s3_expert_system.py)
* [Script 4: Evaluator](./demo/s4_predictor.py)
* [Script 5: Boundary fixer](./demo/s5_fix_noize.py)

## Contact

If you have any problem, please email Jialun Niu: `niujialun@tongji.edu.cn`.

### License ###

[MIT License](./LICENSE)