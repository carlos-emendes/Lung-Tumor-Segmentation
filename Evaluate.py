import numpy as np
import torch
from Training import LungSegmentation
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from Dataset import LungDataset
from tqdm import tqdm
from matplotlib.animation import FuncAnimation


model = LungSegmentation.load_from_checkpoint('best_model.ckpt')
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


val_path = Path('Processed/val/')
val_dataset = LungDataset(val_path)


class DiceScore(torch.nn.Module):
    """
    class to compute the Dice Loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        # flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum()  # Counter
        denum = pred.sum() + mask.sum()  # denominator
        dice = (2 * counter) / denum

        return dice


preds = []
labels = []

for slice, label in tqdm(val_dataset):
    slice = torch.tensor(slice).float().to(device).unsqueeze(0)
    with torch.no_grad():
        pred = torch.sigmoid(model(slice))
    preds.append(pred.cpu().numpy())
    labels.append(label)

preds = np.array(preds)
labels = np.array(labels)

dice_score = DiceScore()(torch.from_numpy(preds), torch.from_numpy(labels).unsqueeze(0).float())
print(f"The Val Dice Score is: {dice_score}")

THRESHOLD = 0.5

subject = Path("Task06_Lung/imagesTr/lung_045.nii.gz")
sample_path_label= Path("Task06_Lung/labelsTr/lung_045.nii.gz")

ct = nib.load(subject).get_fdata().astype(np.float64) / 3071  # standardize
ct = ct[:,:,30:]  # crop
label = nib.load(sample_path_label)
mask = label.get_fdata().astype(np.uint8)
mask = mask[:,:,30:]

segmentation = []
scan = []
mask_ = []

for i in range(ct.shape[-1]):
    slice = ct[:, :, i]
    slice = cv2.resize(slice, (256, 256))
    slice = torch.tensor(slice)
    scan.append(slice)
    slice = slice.unsqueeze(0).unsqueeze(0).float().to(device)

    slice_mask = mask[:, :, i]
    slice_mask_resized = cv2.resize(slice_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_.append(slice_mask_resized)

    with torch.no_grad():
        pred = model(slice)[0][0].cpu()
    pred = pred > THRESHOLD
    segmentation.append(pred)


fig, ax = plt.subplots(1, 2)
fig.suptitle("Lung Tumor Detection", fontsize=12, y=0.95, color='gray')
im_ct_1 = ax[0].imshow(scan[0], cmap='gray')
mask_data_0 = np.ma.masked_where(mask_[0] == 0, mask_[0])
im_mask_1 = ax[0].imshow(mask_data_0, cmap='autumn', alpha=0.5)

im_ct_2 = ax[1].imshow(scan[0], cmap='gray')
mask_data_0 = np.ma.masked_where(segmentation[0] == 0, segmentation[0])
im_mask_2 = ax[1].imshow(mask_data_0, cmap='autumn', alpha=0.5)


def update(frame):
    im_ct_1.set_array(scan[frame])
    current_mask = mask_[frame]
    masked_slice = np.ma.masked_where(current_mask == 0, current_mask)
    im_mask_1.set_array(masked_slice)

    im_ct_2.set_array(scan[frame])
    current_mask = segmentation[frame]
    masked_slice = np.ma.masked_where(current_mask == 0, current_mask)
    im_mask_2.set_array(masked_slice)

    ax[0].set_title("Ground Truth", fontsize=14, fontweight='bold')
    ax[0].set_xlabel(f"Axial Slice: {frame}/{len(scan)}", fontsize=10, color='gray')
    ax[1].set_title("U-Net Segmentation", fontsize=14, fontweight='bold')
    ax[1].set_xlabel(f"Axial Slice: {frame}/{len(scan)}", fontsize=10, color='gray')
    return [im_ct_1, im_mask_1, im_ct_2, im_mask_2]


ani = FuncAnimation(fig, update, frames=ct.shape[2], interval=5)
ani.save('ct_with_prediction_id45.gif', fps=10,)
