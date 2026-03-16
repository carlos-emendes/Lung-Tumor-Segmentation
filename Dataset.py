import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import v2 #another options are imgaug and albumentation, however, these versions are generating conflicts. Since the augmentations is simple, I will rely
# on torchvision

class LungDataset(torch.utils.data.Dataset):
    def __init__(self,root,augment=None):
        self.all_files = self.extract_files(root)


        if augment:
            self.augment=augment
        else:
            self.augment= False

    @staticmethod
    def extract_files(root):
        files= []
        for folder in root.glob('*'):
            slice_path = folder/'data'
            for slice_data in slice_path.glob('*.npy'):
                files.append(slice_data)
        return files
    @staticmethod
    def change_img_to_label_path(path):
        parts = list(path.parts)
        parts[parts.index('data')] = 'mask'
        return Path(*parts)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self,idx):
        slice_path = self.all_files[idx]
        mask_path = self.change_img_to_label_path(slice_path)
        slice_data= np.load(slice_path)
        mask = np.load(mask_path)

        if self.augment:
            slice_tensor, mask_tensor = torch.from_numpy(slice_data).unsqueeze(0),torch.from_numpy(mask).unsqueeze(0)
            image_and_mask = torch.cat([slice_tensor,mask_tensor], dim=0)
            image_and_mask= self.augment(image_and_mask)
            # print(image_and_mask.shape)
            slice_tensor = image_and_mask[0,:,:]
            mask_tensor = torch.round(image_and_mask[1,:,:])

            #now, since I still want to work in numpy, we change it back to numpy
            slice_tensor = slice_tensor.numpy()
            mask_tensor = mask_tensor.numpy()
            return np.expand_dims(slice_tensor,0),np.expand_dims(mask_tensor,0)

        else:
            return np.expand_dims(slice_data,0), np.expand_dims(mask,0)
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = Path('Processed/train/')

    augmentation = v2.Compose([
        v2.RandomAffine(degrees=(-45, 45), scale=(0.85, 1.15)),
        v2.ElasticTransform()])

    dataset = LungDataset(path, augmentation)
    # set a condition to visualize the first image with tumor

    data, mask = dataset.__getitem__(125)
    mask_ = np.ma.masked_where(mask == 0, mask)

    fig = plt.figure(figsize=(20 / 2.54, 20 / 2.54))
    plt.imshow(data[0], cmap='gray')
    plt.imshow(mask_[0], alpha=0.5)
    plt.show()
