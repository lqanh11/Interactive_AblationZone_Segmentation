# Automatic ablation zone segmentation
In this folder, we will provide an instruction for automatic ablation zone segmentation using the nnUNet framework:
>Please note that we use the nnUNet v1 and we hope that nnUNet v2 does not have a huge difference. 
## nnUNet installation
You should follow the [Installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) from the original nnUNet repository.
Please make sure that you also set the environment variables by following this [document](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md) from nnUNet.
## Run inference with trained models
You can download the checkpoint from this [link](https://drive.google.com/file/d/1IWOiX4dXbgjeuEHk9XbV40bsqNRuI6CU/view?usp=drive_link)
Please use the following command to get the prediction:
`nnUNet_predict -i {input_folder} -o {output_folder} -t 16 -m 3d_fullres -f 0 -p nnUNetPlans_pretrained_PreTrained --save_npz`