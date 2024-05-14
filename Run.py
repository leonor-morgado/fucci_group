import torch
from torch.utils.data import Dataset, DataLoader
from FUCCIDataLoader import FUCCIDataset
from UnetModel import UNet
from Train import train
from torchvision import transforms
import time


train_dataset = FUCCIDataset(root_dir="leonor", source_channels=0, target_channels=(0, 1), transform=transforms.RandomCrop(256)) #Do we need to crop? Guess also some data augmentation here would be nice
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8) 

save_path = "/group/dl4miacourse/projects/FUCCI/leonor/Models/model1.pt"


### Declare the U-Net and its parameters

model = UNet(depth=2, in_channels=1, out_channels=2)
model_name = 'test1'
#TODO tensoboard logs
#logger = SummaryWriter(f"unet_runs/{model_name}")

optimizer = torch.optim.Adam(model.parameters())
loss_function: torch.nn.Module = torch.nn.MSELoss()
start_time =time.time()
n_epochs = 360
for epoch in range(n_epochs):

    train(model, train_loader, optimizer=optimizer, loss_function=loss_function, epoch=epoch)
    stop_time = time.time()
    if epoch %5 == 0:
        #print(f"{stop_time- start_time:3f}")
        torch.save(model.state_dict(), save_path)
    #with logger train(model, train_loader, optimizer=optimizer, loss_function=loss_function, epoch=epoch, tb_logger=logger)