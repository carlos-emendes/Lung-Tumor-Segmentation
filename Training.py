from Dataset import LungDataset
from Model import Unet
import lightning as L
from torchvision.transforms import v2
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

'''This class is responsible for training and evaluating the Machine Learning.
The system is very structured and has required functions:
forward, training_step,validation_step, configure_optimizers.
'''
class LungSegmentation(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model= Unet()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr= 1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self,data):
        return self.model(data)
    
    def training_step(self,batch,batch_idx):
        ct_data,mask = batch
        mask= mask.float()
        pred= self(ct_data.float())
        loss = self.loss_fn(pred,mask)
        self.log('Train Loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        ct_data,mask = batch
        mask= mask.float()
        pred= self(ct_data.float())
        loss = self.loss_fn(pred,mask)
        self.log('Val Loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return [self.optimizer]

if __name__ == '__main__':
    train_path = Path('Processed/train/')
    val_path = Path('Processed/val/')
    
    # The following augmentation has 45 degrees of rotation, scale 0.85 to 1.15 and translation in 15%
    augmentation = v2.Compose([
                v2.RandomAffine(degrees=(-45,45),scale=(0.85,1.15),translate=(0.15,0.15)),
                v2.ElasticTransform()])
    
    # We define the train and validation set
    train_dataset = LungDataset(train_path)
    val_dataset = LungDataset(val_path)
    
    '''This part is necessary for oversampling. Since the dataset has huge amount of slices without tumor, we have an imbalance with data.
    Hence, we must apply different weight for the slices that has tumor.
    '''
    
    target_list = []
    for _, label in tqdm(train_dataset):
        # Check if mask contains a tumorous pixel:
        if np.any(label):
            target_list.append(1)
        else:
            target_list.append(0)
            
    # check which mask has tumor and apply the weight on thoses masks.
    uniques = np.unique(target_list, return_counts=True)
    fraction = uniques[1][0] / uniques[1][1]
    weight_list = []
    for target in target_list:
        if target == 0:
            weight_list.append(1)
        else:
            weight_list.append(fraction)
            
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))
    #this are variables based on your computer.
    batch_size = 8 
    num_workers = 4
    #Notice we set the sampler only on the training set because the validation should not rely on weight information, we want real a validation data.
    train_loader = torch.utils.data.DataLoader (train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    val_loader = torch.utils.data.DataLoader (val_dataset, batch_size=batch_size,shuffle=False, num_workers=num_workers)

    model = LungSegmentation() #We initialize the Segmentation Class.
    # model = LungSegmentation.load_from_checkpoint('best_model.ckpt')
    #  if we want to start from a loaded model, we can use the following : model = LungSegmentation.load_from_checkpoint('your_saved_model.ckpt')
    
    checkpoint_callback = ModelCheckpoint(monitor="Val Loss", save_top_k = 20, mode='min')
    gpu = 1 #Another parameter based on the computer
    trainer = L.Trainer(devices =gpu, logger = TensorBoardLogger(save_dir='./logs'), log_every_n_steps=1,callbacks=checkpoint_callback, max_epochs = 20, num_sanity_val_steps=0)
    #Here we start the training.
    torch.set_float32_matmul_precision('high')
    trainer.fit(model,train_loader,val_loader)