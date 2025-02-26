# PVModuleAnalysis  
Photovoltaic module analysis done with machine learning at UCF for senior design in 2022-2023, as a collaborative project with classmates.

To run the test data set warped notebook, you will need to have `panel_model.pth` to be able to load the weights and run it in a Google Colab notebook. Furthermore, you will need to create a folder called `temp_folder2` in `/content` on Google Colab so it can run smoothly and save the images to that folder.

## To run Python executable:
- Clone branch `convert-to-library` and run the executable:  
    - `python PV-UVF-Correction.py`  
        - The user may pass `-w` or `--warp_module` to wrap the PV module, and base the cell segmentation and indexing on the warped image instead of the cropped image.  
    - Select a folder filled with UVF images.  
- A new folder in your working directory will be created labeled `uvf_correction_folder` filled with images of the corrected solar panels and folders of their corresponding indexed cells.
