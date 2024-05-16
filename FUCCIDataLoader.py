from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
from aicsimageio import AICSImage


class FUCCIDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir,
                 source_channels: tuple, target_channels: tuple, transform=None, img_transform=None):
        # TODO enable different file edings for source and target
        self.root_dir = "/group/dl4miacourse/projects/FUCCI/leonor/Toxo" # the directory with all the training samples
        self.video_files = os.listdir(os.path.join(self.root_dir, "Source"))  # list the videos
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.source_channels = source_channels
        self.target_channels = target_channels

        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        self.inp_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Normalize([0.0], [1.0]),  # 0 = mean and 1 = variance
                
               
            ]
        )

        self.open_videos = []
        # we use a list to support videos of varying length
        self.frames_per_video = []

        # same for masks
        self.open_targets = []

        self.source_mean = []
        self.source_std = []
        self.target_mean = []
        self.target_std = []

        for video_file_base in self.video_files:
            video_file = os.path.join(self.root_dir, "Source", video_file_base)
            target_file = os.path.join(self.root_dir, "Target", video_file_base)
           
            video = AICSImage(video_file)
            target = AICSImage(target_file)
          
            n_frames_source = video.dims.T
            n_frames_target = target.dims.T
            if not n_frames_target == n_frames_source:
                raise ValueError(f"Video {video_file_base} does not have "
                                  "the same frames in target and source")
           
            video_mean = video.data.reshape(video.data.shape[0], -1).mean(axis=1)
            video_std = video.data.reshape(video.data.shape[0], -1).std(axis=1)
            target_mean = target.data.reshape(video.data.shape[0], video.data.shape[1], -1).mean(axis=2)
            target_std = target.data.reshape(video.data.shape[0], video.data.shape[1],-1).std(axis=2)
            
            self.source_mean.append(video_mean)
            self.source_std.append(video_std)

            self.target_mean.append(target_mean)
            self.target_std.append(target_std)

            self.open_videos.append(video)
            self.open_targets.append(target)
            self.frames_per_video.append(n_frames_source)


        print("anything")



    # get the total number of samples
    def __len__(self):
        return sum(self.frames_per_video)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # to determine from which file to read
        # TODO implement check
        video_idx = -1
        frame_idx = -1

        frames_seen = 0
        for i, frames in enumerate(self.frames_per_video):
            frames_seen += frames
            if idx < frames_seen:
                video_idx = i
                frame_idx = idx - (frames_seen - frames)
                break
    
        # TODO wrap return_dims in functions
        # if frame_idx < 0:
            # print("something")

        return_dims = "CYX"
       
        source_frames = self.open_videos[video_idx].get_image_dask_data(return_dims, C=self.source_channels, T=frame_idx).astype(float)

        target_frames = self.open_targets[video_idx].get_image_dask_data(return_dims, C=self.target_channels, T=frame_idx).astype(float)

        mean_source = self.source_mean[video_idx][frame_idx]
        std_source = self.source_std[video_idx][frame_idx]
        mean_target = self.target_mean[video_idx][frame_idx]
        std_target = self.target_std[video_idx][frame_idx]
        
        
        source_frames = torch.from_numpy(source_frames.compute())
        target_frames = torch.from_numpy(target_frames.compute())
        
        source_frames = (source_frames - mean_source)/std_source
        target_frames = (target_frames - mean_target[:,None,None])/std_target[:, None, None]


        if self.transform is not None:
            
            seed = torch.seed()
            torch.manual_seed(seed)

            source_frames = self.transform(source_frames)
            torch.manual_seed(seed)
            target_frames = self.transform(target_frames)

            ## mine 

            # crop_std_ch1_thr = 0.47
            # crop_std_ch2_thr = 0.38
            # batch_found = False
            # max_iter_limit = 100
            # cur_iter = 0 
            # while not batch_found:
            #     seed = torch.seed()
            #     torch.manual_seed(seed)
            #     cur_iter += 1
            #     source_frames = self.transform(source_frames)

            #     torch.manual_seed(seed)
            #     target_frames = self.transform(target_frames)
            #     std_ch1 = target_frames[0].std()
            #     std_ch2 = target_frames[1].std()
            #     if cur_iter > max_iter_limit:
            #         batch_found = True
            #         # print("compromize")

            #     if std_ch1 > crop_std_ch1_thr:
            #         batch_found = True
            #     if std_ch2 > crop_std_ch2_thr:
            #         batch_found = True
                    # print

            # print("batch found")


        if self.img_transform is not None:
            source_frames = self.img_transform(source_frames)
        return source_frames.float(), target_frames.float()

        
        
'''
def load_nd2_file(nd2_file):
    """Load the file (TODO flat field correction?)"""

    return loaded_file
'''