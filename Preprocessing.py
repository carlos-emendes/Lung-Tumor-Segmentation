
from pathlib import Path
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
''''
pathlib : easy way to define paths.
nibabel : reads .nii file
cv2 : perform resize of dataset (512,512) -> (256,256)
'''
root = Path('Task06_Lung/imagesTr/')
all_files = list(root.glob("lung_*"))  # Get all subjects
SAVE_PATH= Path('Processed')

# Develop an easy way to find the masks in the folder. Since the data and mask has same name, we just
# need to change the folder location Imagestr to labelsTr
def change_img_to_label(path):
    parts = list(path.parts)
    parts[parts.index('imagesTr')] = 'labelsTr'
    return Path(*parts)


for counter, path_to_mri_data in enumerate(tqdm(all_files)):

    path_label = change_img_to_label(path_to_mri_data) #set the masks location

    data = nib.load(path_to_mri_data)
    label = nib.load(path_label)
    assert nib.aff2axcodes(data.affine) == ('L', 'A', 'S') #this is to verify if orientation of files are correct

    #get_fdata() loads the 3D array
    mri = data.get_fdata().astype(np.float64)
    mask = label.get_fdata().astype(np.uint8)

    ''''Here is the preprocessing part. The normalization for CT images are quite simple. 
    no mean normalization is applied. We only divide everything to 3071. Also, for the specific task, we want to
    crop the lower abdomen part (first 30 slices). Lastly, the last 6 dataset will be set as validation.
    '''
    mri = mri[:, :, 30:] / 3071
    mask = mask[:, :, 30:]

    train_or_val = 'train' if (counter < len(all_files) - 6) else 'val'

    # now we need to save all slices separated
    current_path = SAVE_PATH / train_or_val / str(counter)
    #Since 3D Unet is quite cost, we will save the slices separately
    for i in range(mri.shape[-1]):
        # get a single slice for MRI and Mask
        slice_mri = mri[:, :, i]
        slice_mask = mask[:, :, i]
        '''This is the last stage of preprocessing, we change the size of the dataset to (256,256), for the mask,
         we apply inter_nearest'''
        slice_mri_resized = cv2.resize(slice_mri, (256, 256))
        slice_mask_resized = cv2.resize(slice_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # save MRI and mask in a different folder
        slice_path = current_path / 'data'
        mask_path = current_path / 'mask'
        slice_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)

        np.save(slice_path / str(i), slice_mri_resized)
        np.save(mask_path / str(i), slice_mask_resized)


