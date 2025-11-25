# EKRM: Expert K-Means Reconstruction Method

This repository contains the implementation of the Expert K-Means Reconstruction Method (EKRM), a novel approach for mesostructure image reconstruction.

The method is specifically designed to facilitate the reconstruction of high-resolution mesostructural images, with an example using granite as the target material.

The algorithm is detailed in the paper titled "Expert K-Means Reconstruction Method: a novel image processing approach for mesostructure reconstruction of crystalline rocks", which is currently under review. Updates regarding the paper's publication status will be provided in due course.

Please cite this software as:

```text
Haoyu Pan, Cheng Zhao, Jialun Niu, Jinquan Xing, Huiguan Chen, Rui Zhang. Expert K-Means Reconstruction Method: a novel image processing approach for mesostructure reconstruction of crystalline rocks.
```

## Requirements

To run this code, the following environment is required:

- Python: 3.10
- CUDA: 12.9

You can install the necessary Python dependencies by running the following command:

```shell
pip install -r requirements.txt
```

## Usage

A [demo directory](./demo) is provided, which includes both the source image and the corresponding reconstruction result. The directory contains the following files:

* [Source optical image](./demo/befast-granite.jpg)
* [Reconstruction result](./demo/befast-granite-result.png)
* [Script 1: Config](./demo/s1_config.py)
* [Script 2: Trainer](./demo/s2_trainer.py)
* [Script 3: Expert system](./demo/s3_expert_system.py)
* [Script 4: Predictor](./demo/s4_predictor.py)
* [Script 5: Boundary fixer](./demo/s5_boundary_fixer.py)
* [Script 6: Batch process](./demo/s6_batch_process.py)

### Steps for Reconstruction

1. Configuration: Start by adjusting the configuration settings in Script 1 (Config) to suit your input image.
   
2. Training: Run Script 2 (Trainer) and Script 3 (Expert System) sequentially. After execution, a Windows Explorer window will pop up, prompting you to manually classify the images into the corresponding folders. This step is essential for integrating the expert system's knowledge into the reconstruction process.

3. Post-Processing: Once all images are correctly classified into their respective folders, run Script 4 (Predictor) and Script 5 (Boundary Fixer). This will generate the final reconstructed output.

### Output Files

Upon successful execution, the output files will be saved to the following path:

```text
./demo/output/befast-granite.bmp/LAB-cluster-100/240506-153506/fix-noize-initial-convolve-5-pixel-1250/fixed.png
```

Additionally, a user-friendly numpy array version of the reconstruction result will be generated at:

```text
./demo/output/befast-granite.bmp/LAB-cluster-100/240506-153506/fix-noize-initial-convolve-5-pixel-1250/fixed.npz
```

These output files include both visual results and data in a format suitable for further analysis.

### Training Weights and Classification Results

The training weights and classification results for the provided example images are also available in the output folder for reference.

## Contact

For any issues or inquiries, please contact Jialun Niu at:  
Email: [niujialun@tongji.edu.cn](mailto:niujialun@tongji.edu.cn)

## License

This project is licensed under the [MIT License](./LICENSE).
