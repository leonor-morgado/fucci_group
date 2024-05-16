import torch
from torch.utils.data import Dataset, DataLoader
from FUCCIDataLoader import FUCCIDataset
from UnetModel import UNet
from Train import train
from torchvision import transforms
import time

import wandb
import random

# start a new wandb run to track this script

# sim
    # log metrics to wandb

# [optional] finish the wandb run, necessary in notebooks

train_dataset = FUCCIDataset(root_dir="leonor", source_channels=0, target_channels=0, transform=transforms.RandomCrop(256)) #Do we need to crop? Guess also some data augmentation here would be nice
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8) 

save_path = "/group/dl4miacourse/projects/FUCCI/leonor/Models/model_toxo.pt"


### Declare the U-Net and its parameters

model = UNet(depth=3, in_channels=1, out_channels=1 )
model_name = 'model2'
#TODO tensoboard logs
#logger = SummaryWriter(f"unet_runs/{model_name}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss_function: torch.nn.Module = torch.nn.MSELoss()
start_time =time.time()
n_epochs = 2000
wandb.init(
    # set the wandb project where this run will be logged
    project="toxo",

    # track hyperparameters and run metadata
    config={
    "epochs": n_epochs,
    }
)

for epoch in range(n_epochs):

    train(model, train_loader, optimizer=optimizer, loss_function=loss_function, epoch=epoch)
    stop_time = time.time()
    if epoch %5 == 0:
        #print(f"{stop_time- start_time:3f}")
        torch.save(model.state_dict(), save_path)
    #with logger train(model, train_loader, optimizer=optimizer, loss_function=loss_function, epoch=epoch, tb_logger=logger)
        
#After confirming that runs at least one epoch, go to terminal and nvdia-smi to be sure it's running in the GPU