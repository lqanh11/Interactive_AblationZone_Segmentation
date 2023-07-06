import os
import subprocess
import shutil

import os
import SimpleITK as sitk
import shutil

import os
import numpy as np
import fnmatch
import SimpleITK as sitk
import pickle

def create_mask_from_nii(seg_file_path, save_folder_path):
    
    image_file_name = os.path.basename(seg_file_path)

    namebase = image_file_name.split('.')
    file_name = namebase[0]
    print(file_name)
    
    save_path = os.path.join(save_folder_path, file_name)

    if os.path.exists(save_path) != True:
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    
    seg_itk = sitk.ReadImage(seg_file_path)

    for slice_index in range(seg_itk.GetDepth()):
        save_mask_file_path = os.path.join(save_path, file_name + '-' + format(slice_index, '04d') + '.nii.gz')
        
        seg_slice_itk = seg_itk[:,:,slice_index]
        
        sitk.WriteImage(seg_slice_itk, save_mask_file_path)

    return seg_itk

def get_prob_from_npz(image_namebase, pred_path):


    npz_path = os.path.join(pred_path, image_namebase + '.npz')
    pkl_path = os.path.join(pred_path, image_namebase + '.pkl')

    with open(pkl_path, 'rb') as f:
            data_pkl = pickle.load(f)
            if len(data_pkl) < 9:
                data_pkl = data_pkl[0]

    original_size = data_pkl['original_size_of_raw_data']
    original_spacing = data_pkl['itk_spacing']
    original_origin = data_pkl['itk_origin']

    crop_bbox = data_pkl['crop_bbox']
    data_prob = np.load(npz_path)['softmax']
    ablation_prob = data_prob[1].astype(float)
    itk_ablation_prob = sitk.GetImageFromArray(ablation_prob)
    itk_ablation_prob.SetSpacing((original_spacing[2], original_spacing[1], original_spacing[0]))
    itk_image = sitk.Image(int(original_size[2]), int(original_size[1]), int(original_size[0]), sitk.sitkFloat64)
    itk_image.SetSpacing(original_spacing)
    itk_image[crop_bbox[2][0]:crop_bbox[2][1], crop_bbox[1][0]:crop_bbox[1][1], crop_bbox[0][0]:crop_bbox[0][1]] = itk_ablation_prob
    itk_image.SetOrigin(original_origin)

    sitk.WriteImage(itk_image, os.path.join(pred_path, image_namebase + '.nii.gz'))

def getSegmentation(ct_image_path, save_mask_path):
    ## file name extraction
    image_file_name = os.path.basename(ct_image_path)
    image_namebase = image_file_name.split('.')[0]
    
    temp_path = './automatic_segmentation/nnUNet_Temp'
    if os.path.exists(temp_path)!=True:
        os.mkdir(temp_path)

    decathlon_save_path = os.path.join(temp_path, 'decathlon')
    if os.path.exists(decathlon_save_path):
        shutil.rmtree(decathlon_save_path)
        os.mkdir(decathlon_save_path)
    
    decathlon_file_path = os.path.join(decathlon_save_path, image_namebase + '_0000.nii.gz')
    print(image_file_name, decathlon_file_path)

    shutil.copy(ct_image_path, decathlon_file_path)

    save_nnUNet_predict = os.path.join(temp_path, 'test_results_finetuning_1_prob')

    decathlon_file_path_abs =  os.path.abspath(decathlon_save_path)
    save_nnUNet_predict_abs = os.path.abspath(save_nnUNet_predict)

    print(subprocess.getstatusoutput(f'nnUNet_predict -i {decathlon_file_path_abs} -o {save_nnUNet_predict_abs} -m 3d_fullres -t 16 -f 4 -p nnUNetPlans_pretrained_PreTrained --save_npz'))

    get_prob_from_npz(image_namebase, save_nnUNet_predict_abs)

    itk_seg = create_mask_from_nii(os.path.join(save_nnUNet_predict, image_namebase + '.nii.gz'), save_mask_path)

    return itk_seg
