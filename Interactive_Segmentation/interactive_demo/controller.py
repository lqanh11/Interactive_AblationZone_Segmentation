import torch
import numpy as np
from tkinter import messagebox
import cv2
import copy

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks
import SimpleITK as sitk


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5, upperThresh=200, lowerThresh=-100, roi_size=10):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.upperThresh = upperThresh
        self.lowerThresh = lowerThresh

        self.roi_size = roi_size

        self.current_click_state = []

        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        # self.probs_history.append((self._init_mask, self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        # print(x,y)

        self.current_click_state = x,y,is_positive
        

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        # print(pred.shape)

        # itk_pred = sitk.GetImageFromArray(pred)
        # sitk.WriteImage(itk_pred, 'D:/Project/Interactive_Segmentation/ritm_interactive_segmentation/pred_prob.nii.gz')

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            
            clipped_img_array = copy.deepcopy(self.image)
            clipped_img_array[clipped_img_array > self.upperThresh] = self.upperThresh
            clipped_img_array[clipped_img_array < self.lowerThresh] = self.lowerThresh
            rescaled_img_array = 255*(clipped_img_array - self.lowerThresh).astype(np.float)/(self.upperThresh - self.lowerThresh)
            rescaled_img_array = rescaled_img_array.astype(np.uint8)
            
            self.predictor.set_input_image(cv2.cvtColor(rescaled_img_array, cv2.COLOR_BGR2RGB))

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        mode = "RITM"
        if mode == "RITM":
            if self.probs_history:
                current_prob_total, current_prob_additive = self.probs_history[-1]
                return  np.maximum(current_prob_total, current_prob_additive)
            else:
                return None
        elif mode == "Proposed":

            if self.probs_history:
                current_prob_total, current_prob_additive = self.probs_history[-1]


                if len(self.probs_history) > 1:
                    current_prob_total_previous, current_prob_additive_previous = self.probs_history[-2]

                    current_prob_additive_roi = current_prob_additive_previous.copy()

                    x,y,is_positive = self.current_click_state

                    roi_pred = np.zeros_like(current_prob_additive)
                    roi_pred[y-self.roi_size:y+self.roi_size,x-self.roi_size:x+self.roi_size] = current_prob_additive[y-self.roi_size:y+self.roi_size,x-self.roi_size:x+self.roi_size]

                    # itk_roi_pred = sitk.GetImageFromArray(roi_pred)
                    # sitk.WriteImage(itk_roi_pred, 'D:/Project/Interactive_Segmentation/ritm_interactive_segmentation/roi_pred.nii.gz')

                    # itk_current_add = sitk.GetImageFromArray(current_prob_additive)
                    # sitk.WriteImage(itk_current_add, 'D:/Project/Interactive_Segmentation/ritm_interactive_segmentation/curr_add_prob.nii.gz')
                    # print(self.roi_size)
                    if self.roi_size != 0:
                        if is_positive:
                            current_prob_additive_roi[y-self.roi_size:y+self.roi_size,x-self.roi_size:x+self.roi_size] = np.maximum(current_prob_additive_roi[y-self.roi_size:y+self.roi_size,x-self.roi_size:x+self.roi_size], roi_pred[y-self.roi_size:y+self.roi_size,x-self.roi_size:x+self.roi_size])
                            
                            self.probs_history[-1] = current_prob_total, current_prob_additive_roi
                            
                            return current_prob_additive_roi
                        else:
                            current_prob_additive_roi[y-self.roi_size:y+self.roi_size,x-self.roi_size:x+self.roi_size] = np.minimum(current_prob_additive_roi[y-self.roi_size:y+self.roi_size,x-self.roi_size:x+self.roi_size], roi_pred[y-self.roi_size:y+self.roi_size,x-self.roi_size:x+self.roi_size])
                            
                            self.probs_history[-1] = self.probs_history[-1][0], current_prob_additive_roi
                            
                            return current_prob_additive_roi
                    else:
                        return np.maximum(current_prob_total, current_prob_additive)
                
                else:
                    return np.maximum(current_prob_total, current_prob_additive)
            else:
                return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            # print(np.shape(self.current_object_prob))

            # itk_pro = sitk.GetImageFromArray(self.current_object_prob)
            # sitk.WriteImage(itk_pro, 'D:/Project/Interactive_Segmentation/ritm_interactive_segmentation/test_image_prob.nii.gz')


            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask
    @property
    def label_mask(self):
        return self.label.copy()


    def get_visualization(self, alpha_blend, click_radius, upperThrsh, lowerThrsh):
        if self.image is None:
            return None

        self.upperThresh = upperThrsh
        self.lowerThresh = lowerThrsh

        clipped_img_array = copy.deepcopy(self.image)
        clipped_img_array[clipped_img_array > self.upperThresh] = self.upperThresh
        clipped_img_array[clipped_img_array < self.lowerThresh] = self.lowerThresh
        rescaled_img_array = 255*(clipped_img_array - self.lowerThresh).astype(np.float)/(self.upperThresh - self.lowerThresh)
        rescaled_img_array = rescaled_img_array.astype(np.uint8)

        results_mask_for_vis = self.result_mask
        vis = draw_with_blend_and_clicks(cv2.cvtColor(rescaled_img_array, cv2.COLOR_BGR2RGB), mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        return vis

    def update_roi_size(self, value):
        self.roi_size = value

