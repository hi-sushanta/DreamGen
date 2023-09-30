import torch
import torchvision
from torchvision import transforms
from transformers import T5Tokenizer
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.dataset import CustomDataset
import pandas as pd
from model.Network import Generator,Discriminator
from torch import optim 
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

class Utils():
    def __init__(self):
        pass
    def gat_dataloader(self,base_dir,data_filepath,target_size):
        pd_df = pd.read_excel(data_filepath)
        pd_df.drop("comment_number",axis=1,inplace=True)
        image_with_text = pd_df.to_numpy().squeeze()
        dataset = CustomDataset(base_dir,image_with_text,target_size)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=4)
        return dataloader

    def get_model(self,image_channels,output_channel,embedding_dim,
                        image_size, learning_rate, beta_value,
                        device='cpu', if_pretrain=False,gan_path=None,disc_path=None):
        
        gan = nn.DataParallel(Generator(image_channels,output_channel,embedding_dim,image_size).to(device))
        disc = nn.DataParallel(Discriminator().to(device))

        # Create the optimizer with Adam (or any other optimizer you prefer)
        g_optim = optim.Adam(gan.parameters(), lr=learning_rate,betas=beta_value)
        d_optim = optim.Adam(disc.parameters(),lr=0.0002,betas=beta_value)
        
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)

        if if_pretrain:
            print("Using pretrained model")
            gan_statedict = torch.load("/content/drive/MyDrive/DL-Project/Text-To-Image/DGenerator.pth")
            gan.load_state_dict(gan_statedict)
            disc_statedict = torch.load('/content/drive/MyDrive/DL-Project/Text-To-Image/DDisc.pth')
            disc.load_state_dict(disc_statedict)
        else:
            gan = gan.apply(weights_init)
            disc = disc.apply(weights_init)
        return gan,disc,g_optim,d_optim
    

    def show_tensor_images_train(self,image_tensor,text, num_images=1, size=(3, 256, 256)):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
        '''
        batch_of_image = image_tensor

        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        # create figure
        fig = plt.figure(figsize=(10, 7))

        # setting values to rows and column variables
        rows = 2
        columns = 2


        for i,x in enumerate(batch_of_image):
            # Adds a subplot at the staped position position
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(reverse_transforms(x))
            plt.title(f"{text[i]} {i}")
            plt.axis('off')
        plt.show() # now time to showit
   



