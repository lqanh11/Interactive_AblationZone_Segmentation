# Automatic Ablation Zone Segmentation

In this folder, we provide instructions for performing automatic ablation zone segmentation using the nnUNet framework.

> Please note that we use nnUNet v1, and we expect that nnUNet v2 does not have significant differences.

## nnUNet installation

Please follow the [Installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) from the original nnUNet repository.

Make sure to set the environment variables as well by referring to this [document](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md) provided by nnUNet.

## Run inference with trained models

You can download the checkpoint from this [link](https://drive.google.com/file/d/1IWOiX4dXbgjeuEHk9XbV40bsqNRuI6CU/view?usp=drive_link)

To obtain the prediction, please use the following command:

`nnUNet_predict -i {input_folder} -o {output_folder} -t 16 -m 3d_fullres -f 0 -p nnUNetPlans_pretrained_PreTrained --save_npz`

Follow the provided instructions to install nnUNet and run inference using the trained models effectively.