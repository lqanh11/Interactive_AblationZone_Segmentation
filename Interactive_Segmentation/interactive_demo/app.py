import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import os
import cv2
import numpy as np
from PIL import Image
import SimpleITK as sitk

from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame

from automatic_segmentation.nnUNet import getSegmentation
import time
import pandas as pd
import numpy as np
import copy


class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, model):
        super().__init__(master)
        self.master = master
        master.title("Ablation Zone Segmentation")
        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = args.limit_longest_size

        self.ct_path = None
        self.save_mask_path = None
        self.image_namebase = None

        self.CTimage = sitk.Image(512, 512, 60, sitk.sitkInt16)


        self.currentSlice_index = round(self.CTimage.GetDepth() / 2)
        self.upper_thr = 200
        self.lower_thr = -100


        self.start_interactive_segmetation_time = 0
        self.total_interactive_segmetation_time = 0

        self.start_automatic_segmentation_time = 0
        self.total_automatic_segmentation_time = 0


        self.save_folder_path = './datasets/RFA_AblationZone/Mask_AblationZone'

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image,
                                                upperThresh=self.upper_thr,
                                                lowerThresh=self.lower_thr,
                                                )


        self._init_state()
        self._add_menu()
        self._add_canvas()
        self._add_directory()
        self._add_buttons()
        
        master.bind('<Control-Button-5>', self._load_new_slice_callback)
        master.bind('<Control-Button-4>', self._load_new_slice_callback)
        master.bind('<Control-MouseWheel>', self._load_new_slice_callback)
        master.bind('<Control-z>', self.controller.undo_click)
        # master.bind('<space>', lambda event: self.controller.finish_object())
        # master.bind('a', lambda event: self.controller.partially_finish_object())

        self.state['zoomin_params']['skip_clicks'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['target_size'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w', callback=self._reset_predictor)
        self.state['predictor_params']['net_clicks_limit'].trace(mode='w', callback=self._change_brs_mode)
        self.state['lbfgs_max_iters'].trace(mode='w', callback=self._change_brs_mode)
        self._change_brs_mode()

    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'fixed_crop': tk.BooleanVar(value=True),
                'skip_clicks': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=min(400, self.limit_longest_size)),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },

            'predictor_params': {
                'net_clicks_limit': tk.IntVar(value=8)
            },
            'brs_mode': tk.StringVar(value='NoBRS'),
            'prob_thresh': tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20),

            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=1),
            'lower_thresh': tk.IntVar(value=-100),
            'upper_thresh': tk.IntVar(value=200),

            'roi_size': tk.IntVar(value=10),
        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill='x')

        button = FocusButton(self.menubar, text='Load image', command=self._load_image_callback)
        button.pack(side=tk.LEFT)
        self.save_mask_btn = FocusButton(self.menubar, text='Save mask', command=self._save_mask_callback)
        self.save_mask_btn.pack(side=tk.LEFT)
        self.save_mask_btn.configure(state=tk.DISABLED)

        self.save_label_btn = FocusButton(self.menubar, text='Save label', command=self._save_label_callback)
        self.save_label_btn.pack(side=tk.LEFT)
        self.save_label_btn.configure(state=tk.DISABLED)

        self.load_mask_btn = FocusButton(self.menubar, text='Load mask', command=self._load_mask_callback)
        self.load_mask_btn.pack(side=tk.LEFT)
        self.load_mask_btn.configure(state=tk.DISABLED)

        button = FocusButton(self.menubar, text='About', command=self._about_callback)
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text='Exit', command=self.master.quit)
        button.pack(side=tk.LEFT)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_directory(self):
        self.directory_frame = FocusLabelFrame(self, text="Directory management")
        self.directory_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.directory_frame

        self.directory_options_frame = FocusLabelFrame(master, text="File name")
        self.directory_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        self.textboxFileName = tk.Text(self.directory_options_frame, width=50,height=1)
        self.textboxFileName.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.control_frame

        ##
        self.slices_options_frame = FocusLabelFrame(master, text="Slice management")
        self.slices_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.textboxSliceIndex = tk.Text(self.slices_options_frame, width=10,height=1)
        self.textboxSliceIndex.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.previous_slice_button = \
            FocusButton(self.slices_options_frame, text='Previous\nslice', bg='#b6d7a8', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._load_previous_slice_callback)
        self.previous_slice_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        
        self.next_slice_button = \
            FocusButton(self.slices_options_frame, text='Next\nslice', bg='#ffe599', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._load_next_slice_callback)
        self.next_slice_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        

        ## 
        # self.automatic_options_frame = FocusLabelFrame(master, text="Automatic segmentation")
        # self.automatic_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        # self.automatic_seg_button = \
        #     FocusButton(self.automatic_options_frame, text='AutoSeg', bg='#b6d7a8', fg='black', width=10, height=2,
        #                 state=tk.DISABLED, command=self._nnUNet_seg)
        # self.automatic_seg_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        ##
        self.clicks_options_frame = FocusLabelFrame(master, text="Clicks management")
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.automatic_seg_button = \
            FocusButton(self.clicks_options_frame, text='AutoSeg', bg='#b6d7a8', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._nnUNet_seg)
        self.automatic_seg_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        
        self.finish_object_button = \
            FocusButton(self.clicks_options_frame, text='Finish\nobject', bg='#b6d7a8', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.finish_object)
        self.finish_object_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.undo_click_button = \
            FocusButton(self.clicks_options_frame, text='Undo click', bg='#ffe599', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.undo_click)
        self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_clicks_button = \
            FocusButton(self.clicks_options_frame, text='Reset clicks', bg='#ea9999', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._reset_last_object)
        self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        # self.zoomin_options_frame = FocusLabelFrame(master, text="ZoomIn options")
        # self.zoomin_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        # FocusCheckButton(self.zoomin_options_frame, text='Use ZoomIn', command=self._reset_predictor,
        #                  variable=self.state['zoomin_params']['use_zoom_in']).grid(row=0, column=0, padx=10)
        # FocusCheckButton(self.zoomin_options_frame, text='Fixed crop', command=self._reset_predictor,
        #                  variable=self.state['zoomin_params']['fixed_crop']).grid(row=1, column=0, padx=10)
        # tk.Label(self.zoomin_options_frame, text="Skip clicks").grid(row=0, column=1, pady=1, sticky='e')
        # tk.Label(self.zoomin_options_frame, text="Target size").grid(row=1, column=1, pady=1, sticky='e')
        # tk.Label(self.zoomin_options_frame, text="Expand ratio").grid(row=2, column=1, pady=1, sticky='e')
        # BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['skip_clicks'],
        #                       min_value=-1, max_value=None, vartype=int,
        #                       name='zoom_in_skip_clicks').grid(row=0, column=2, padx=10, pady=1, sticky='w')
        # BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['target_size'],
        #                       min_value=100, max_value=self.limit_longest_size, vartype=int,
        #                       name='zoom_in_target_size').grid(row=1, column=2, padx=10, pady=1, sticky='w')
        # BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['expansion_ratio'],
        #                       min_value=1.0, max_value=2.0, vartype=float,
        #                       name='zoom_in_expansion_ratio').grid(row=2, column=2, padx=10, pady=1, sticky='w')
        # self.zoomin_options_frame.columnconfigure((0, 1, 2), weight=1)

        # self.brs_options_frame = FocusLabelFrame(master, text="BRS options")
        # self.brs_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        # menu = tk.OptionMenu(self.brs_options_frame, self.state['brs_mode'],
        #                      *self.brs_modes, command=self._change_brs_mode)
        # menu.config(width=11)
        # menu.grid(rowspan=2, column=0, padx=10)
        # self.net_clicks_label = tk.Label(self.brs_options_frame, text="Network clicks")
        # self.net_clicks_label.grid(row=0, column=1, pady=2, sticky='e')
        # self.net_clicks_entry = BoundedNumericalEntry(self.brs_options_frame,
        #                                               variable=self.state['predictor_params']['net_clicks_limit'],
        #                                               min_value=0, max_value=None, vartype=int, allow_inf=True,
        #                                               name='net_clicks_limit')
        # self.net_clicks_entry.grid(row=0, column=2, padx=10, pady=2, sticky='w')
        # self.lbfgs_iters_label = tk.Label(self.brs_options_frame, text="L-BFGS\nmax iterations")
        # self.lbfgs_iters_label.grid(row=1, column=1, pady=2, sticky='e')
        # self.lbfgs_iters_entry = BoundedNumericalEntry(self.brs_options_frame, variable=self.state['lbfgs_max_iters'],
        #                                                min_value=1, max_value=1000, vartype=int,
        #                                                name='lbfgs_max_iters')
        # self.lbfgs_iters_entry.grid(row=1, column=2, padx=10, pady=2, sticky='w')
        # self.brs_options_frame.columnconfigure((0, 1), weight=1)

        self.prob_thresh_frame = FocusLabelFrame(master, text="Predictions threshold")
        self.prob_thresh_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.prob_thresh_frame, from_=0.0, to=1.0, command=self._update_prob_thresh,
                             variable=self.state['prob_thresh']).pack(padx=10)

        self.alpha_blend_frame = FocusLabelFrame(master, text="Alpha blending coefficient")
        self.alpha_blend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.alpha_blend_frame, from_=0.0, to=1.0, command=self._update_blend_alpha,
                             variable=self.state['alpha_blend']).pack(padx=10, anchor=tk.CENTER)

        self.click_radius_frame = FocusLabelFrame(master, text="Visualisation click radius")
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.click_radius_frame, from_=0, to=7, resolution=1, command=self._update_click_radius,
                             variable=self.state['click_radius']).pack(padx=10, anchor=tk.CENTER)

        self.roi_thresh = FocusLabelFrame(master, text="ROI Size")
        self.roi_thresh.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.roi_thresh, from_=0, to=50, resolution=1, command=self._update_roi_size,
                             variable=self.state['roi_size']).pack(padx=10, anchor=tk.CENTER)

        self.lower_thresh = FocusLabelFrame(master, text="Lower threshold")
        self.lower_thresh.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.lower_thresh, from_=-1024, to=3072, resolution=1, command=self._update_lower_thresh,
                             variable=self.state['lower_thresh']).pack(padx=10, anchor=tk.CENTER)
        self.upper_thresh = FocusLabelFrame(master, text="Upper threshold")
        self.upper_thresh.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.upper_thresh, from_=-1024, to=3072, resolution=1, command=self._update_upper_thresh,
                             variable=self.state['upper_thresh']).pack(padx=10, anchor=tk.CENTER)


    def _nnUNet_seg(self):
        self.start_automatic_segmentation_time = time.time()

        self.save_folder_path='./automatic_segmentation/nnUNet_mask'
        itk_seg = getSegmentation(self.ct_path, self.save_folder_path)
        
        self.total_automatic_segmentation_time = time.time() - self.start_automatic_segmentation_time

        print("Inference time automatic: ", self.total_automatic_segmentation_time)

        self.save_mask_path = os.path.join(self.save_folder_path, self.image_namebase)

        mask_file_path = os.path.join(self.save_mask_path, self.image_namebase + '-' + format(self.currentSlice_index, '04d') + '.nii.gz')
        
        print(mask_file_path)

        if os.path.exists(mask_file_path):

            itk_mask = sitk.ReadImage(mask_file_path)
            mask = sitk.GetArrayFromImage(itk_mask)
            self.controller.set_mask(mask)
            self._update_image()    
    
    def _load_next_slice_callback(self):
        save_mask_file_path = os.path.join(self.save_mask_path, self.image_namebase + '-' + format(self.currentSlice_index, '04d') + '.nii.gz')
        
        if self._check_entry(self):
            mask = self.controller.current_object_prob
            if mask is None:
                pass
            else:
                seg_slice_itk = sitk.GetImageFromArray(mask) 
                sitk.WriteImage(seg_slice_itk, save_mask_file_path)
                # if np.sum(mask) != 0:
                #     if mask.max() < 256:
                #         mask = mask.astype(np.uint8)
                #         mask *= 255 // mask.max()
                #     cv2.imwrite(save_mask_file_path, mask)

            if self.currentSlice_index < self.CTimage.GetDepth()-1:
                self.currentSlice_index += 1
            else:
                self.currentSlice_index = self.CTimage.GetDepth()-1
            self.textboxSliceIndex.delete('1.0', tk.END)
            self.textboxSliceIndex.insert('1.0', str(self.currentSlice_index))

            image = sitk.GetArrayFromImage(self.CTimage[:,:,self.currentSlice_index])
            self.controller.set_image(image)
            mask_file_path = os.path.join(self.save_mask_path, self.image_namebase + '-' + format(self.currentSlice_index, '04d') + '.nii.gz')
            if os.path.exists(mask_file_path):
                itk_mask = sitk.ReadImage(mask_file_path)
                mask = sitk.GetArrayFromImage(itk_mask)
                if np.sum(mask) != 0:
                    self.controller.set_mask(mask)
                    self._update_image()
    
    def _load_previous_slice_callback(self):
        save_mask_file_path = os.path.join(self.save_mask_path, self.image_namebase + '-' + format(self.currentSlice_index, '04d') + '.nii.gz')

        if self._check_entry(self):
            mask = self.controller.current_object_prob
            if mask is None:
                pass
            else:
                seg_slice_itk = sitk.GetImageFromArray(mask) 
                sitk.WriteImage(seg_slice_itk, save_mask_file_path)
            
            if self.currentSlice_index > 0:
                self.currentSlice_index -= 1
            else:
                self.currentSlice_index = 0
            self.textboxSliceIndex.delete('1.0', tk.END)
            self.textboxSliceIndex.insert('1.0', str(self.currentSlice_index))

            image = sitk.GetArrayFromImage(self.CTimage[:,:,self.currentSlice_index])
            self.controller.set_image(image)
            
            mask_file_path = os.path.join(self.save_mask_path, self.image_namebase + '-' + format(self.currentSlice_index, '04d') + '.nii.gz')
            if os.path.exists(mask_file_path):
                itk_mask = sitk.ReadImage(mask_file_path)
                mask = sitk.GetArrayFromImage(itk_mask)
                if np.sum(mask) != 0:
                    self.controller.set_mask(mask)
                    self._update_image()

    def _load_new_slice_callback(self, event):
        save_mask_file_path = os.path.join(self.save_mask_path, self.image_namebase + '-' + format(self.currentSlice_index, '04d') + '.nii.gz')

        if self._check_entry(self):
            mask = self.controller.current_object_prob
            if mask is None:
                pass
            else:
                seg_slice_itk = sitk.GetImageFromArray(mask) 
                sitk.WriteImage(seg_slice_itk, save_mask_file_path)

            if event.num == 5 or event.delta == -120 or event.delta == 1:  # scroll down, zoom out, smaller
                if self.currentSlice_index > 0:
                    self.currentSlice_index -= 1
                else:
                    self.currentSlice_index = 0
            if event.num == 4 or event.delta == 120 or event.delta == -1:  # scroll up, zoom in, bigger
                if self.currentSlice_index < self.CTimage.GetDepth()-1:
                    self.currentSlice_index += 1
                else:
                    self.currentSlice_index = self.CTimage.GetDepth()-1

            self.textboxSliceIndex.delete('1.0', tk.END)
            self.textboxSliceIndex.insert('1.0', str(self.currentSlice_index))

            image = sitk.GetArrayFromImage(self.CTimage[:,:,self.currentSlice_index])
            self.controller.set_image(image)
            
            mask_file_path = os.path.join(self.save_mask_path, self.image_namebase + '-' + format(self.currentSlice_index, '04d') + '.nii.gz')
            if os.path.exists(mask_file_path):
                itk_mask = sitk.ReadImage(mask_file_path)
                mask = sitk.GetArrayFromImage(itk_mask)
                if np.sum(mask) != 0:
                    self.controller.set_mask(mask)
                    self._update_image()

    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.nii *.nii.gz *.mhd *.tiff *.jpg"),
                ("All files", "*.*"),
            ], title="Chose an image")
            self.ct_path = filename

            if len(filename) > 0:

                image_file_name = os.path.basename(filename)
                self.image_namebase = image_file_name.split('.')[0]
                self.save_mask_path = os.path.join(self.save_folder_path, self.image_namebase)
                if os.path.exists(self.save_mask_path)!=True:
                    os.mkdir(self.save_mask_path)


                itk_image = sitk.ReadImage(filename)

                self.currentSlice_index = round(itk_image.GetDepth() / 2)

                self.CTimage = itk_image


                image = sitk.GetArrayFromImage(self.CTimage[:,:,self.currentSlice_index])

                self.controller.set_image(image)
                self.save_mask_btn.configure(state=tk.NORMAL)
                self.save_label_btn.configure(state=tk.NORMAL)
                self.load_mask_btn.configure(state=tk.NORMAL)
                self.automatic_seg_button.configure(state=tk.NORMAL)
                self.previous_slice_button.configure(state=tk.NORMAL)
                self.next_slice_button.configure(state=tk.NORMAL)
                self.textboxFileName.delete('1.0', tk.END)
                self.textboxFileName.insert('1.0', self.image_namebase)
                self.textboxSliceIndex.delete('1.0', tk.END)
                self.textboxSliceIndex.insert('1.0', str(self.currentSlice_index))

                self.start_interactive_segmetation_time = time.time()
                    
                
    def _save_mask_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            mask = self.controller.result_mask
            if mask is None:
                return

            filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
                ("PNG image", "*.png"),
                ("BMP image", "*.bmp"),
                ("All files", "*.*"),
            ], title="Save the current mask as...")

            if len(filename) > 0:
                if mask.max() < 256:
                    mask = mask.astype(np.uint8)
                    mask *= 255 // mask.max()
                cv2.imwrite(filename, mask)
    
    def _save_label_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):

            self.total_interactive_segmetation_time = time.time() - self.start_interactive_segmetation_time

            print("total segmentation time: ", self.total_interactive_segmetation_time)

            filename = filedialog.asksaveasfilename(parent=self.master, initialfile=f'{self.image_namebase}', filetypes=[
                ("Images", "*.nii *.nii.gz *.mhd *.tiff *.jpg"),
                ("All files", "*.*"),
            ], title="Save the current label as...")

            itk_label = self.convert_list_slice_to_img()
            sitk.WriteImage(itk_label, filename + '.nii.gz')


            
    def convert_list_slice_to_img(self):
        image_list = os.listdir(self.save_mask_path)
        itk_slice = sitk.ReadImage(os.path.join(self.save_mask_path, image_list[0]))
        slice_size = itk_slice.GetSize()

        itk_pred = sitk.Image(slice_size[0], slice_size[1], len(image_list), sitk.sitkFloat64)

        for image_file_name in image_list:
            itk_slice = sitk.ReadImage(os.path.join(self.save_mask_path, image_file_name), sitk.sitkFloat64)
            slice_index = int(image_file_name.split('.')[0][-4:])

            itk_pred[:,:,slice_index] = itk_slice
        return itk_pred

    def _load_mask_callback(self):
        if not self.controller.net.with_prev_mask:
            messagebox.showwarning("Warning", "The current model doesn't support loading external masks. "
                                              "Please use ITER-M models for that purpose.")
            return

        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Binary mask (png, bmp)", "*.png *.bmp"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                mask = cv2.imread(filename)[:, :, 0] > 127
                self.controller.set_mask(mask)
                self._update_image()

    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            "Developed by:",
            "K.Sofiiuk and I. Petrov",
            "The MIT License, 2021"
        ]

        messagebox.showinfo("About Demo", '\n'.join(text))

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    def _update_lower_thresh(self, value):
        self._update_image()
    
    def _update_upper_thresh(self, value):
        self._update_image()

    def _update_roi_size(self, value):
        self.controller.update_roi_size(self.state['roi_size'].get())
        self._update_image()

    def _change_brs_mode(self, *args):
        # if self.state['brs_mode'].get() == 'NoBRS':
        #     self.net_clicks_entry.set('INF')
        #     self.net_clicks_entry.configure(state=tk.DISABLED)
        #     self.net_clicks_label.configure(state=tk.DISABLED)
        #     self.lbfgs_iters_entry.configure(state=tk.DISABLED)
        #     self.lbfgs_iters_label.configure(state=tk.DISABLED)
        # else:
        #     if self.net_clicks_entry.get() == 'INF':
        #         self.net_clicks_entry.set(8)
        #     self.net_clicks_entry.configure(state=tk.NORMAL)
        #     self.net_clicks_label.configure(state=tk.NORMAL)
        #     self.lbfgs_iters_entry.configure(state=tk.NORMAL) 
        #     self.lbfgs_iters_label.configure(state=tk.NORMAL)

        self._reset_predictor()

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = self.state['brs_mode'].get()
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None if brs_mode == 'NoBRS' else self.state['predictor_params']['net_clicks_limit'].get()

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None

        predictor_params = {
            'brs_mode': brs_mode,
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': net_clicks_limit,
                'max_size': self.limit_longest_size
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': self.state['lbfgs_max_iters'].get()}
        }
        self.controller.reset_predictor(predictor_params)

    def _click_callback(self, is_positive, x, y):
        
        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self._check_entry(self):
            self.controller.add_click(x, y, is_positive)

    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get(),
                                                  upperThrsh=self.state['upper_thresh'].get(),
                                                  lowerThrsh=self.state['lower_thresh'].get(),)
        if self.image_on_canvas is None:            
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED
        before_1st_click_state = tk.DISABLED if self.controller.is_incomplete_mask else tk.NORMAL

        self.finish_object_button.configure(state=after_1st_click_state)
        self.undo_click_button.configure(state=after_1st_click_state)
        self.reset_clicks_button.configure(state=after_1st_click_state)
        # self.zoomin_options_frame.set_frame_state(before_1st_click_state)
        # self.brs_options_frame.set_frame_state(before_1st_click_state)

        # if self.state['brs_mode'].get() == 'NoBRS':
        #     self.net_clicks_entry.configure(state=tk.DISABLED)
        #     self.net_clicks_label.configure(state=tk.DISABLED)
        #     self.lbfgs_iters_entry.configure(state=tk.DISABLED)
        #     self.lbfgs_iters_label.configure(state=tk.DISABLED)

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked
