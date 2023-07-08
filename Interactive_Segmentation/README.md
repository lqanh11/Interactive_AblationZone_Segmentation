# Interactive ablation zone segmentation
In this folder, we will provide instructions for interactive ablation zone segmentation using the combination of the nnUNet framework and RITM:

You can see a demo video from this [link](https://drive.google.com/file/d/12IVCkXf9-RO78YgvkLSQ0Bz3iR8ry7za/view) 
```.bash
# This command runs the demo of interacitve ablation zone segmentation
python demo.py
# Please note that this demo have intergrated the nnUNet framework, therefore, it requires a workstation with GPU card.
```
**Controls**:

| Key                                                           | Description                        |
| ------------------------------------------------------------- | ---------------------------------- |
| <kbd>Left Mouse Button</kbd>                                  | Place a positive click             |
| <kbd>Right Mouse Button</kbd>                                 | Place a negative click             |
| <kbd>Ctrl</kbd> + <br> <kbd>Scroll Wheel</kbd>    | Load a new slide
| <kbd>Shift</kbd> + <br> <kbd>Scroll Wheel</kbd>    | Zoom in/out 