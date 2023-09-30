from model import Network
from utilis import Utils
import torch
import torchvision
from IPython.display import clear_output
import torch.nn.functional as F
from torchvision import transforms
from transformers import T5Tokenizer
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import torch.optim as optim
from loss.loss import *

# Hyperparameter
base_dir = "/home/chi/Downloads/flicker-text-to-image/flickr30k_images/flickr30k_images/"
xls_datapath = "/home/chi/Documents/Deep-Learning-Project/Text-To-Image/results.xlsx"
device = "gpu" if torch.cuda.is_available() else "cpu" 
beta_value = (0.5,0.999) # It's control how much the optimizer rememeber previous gradient.
learning_rate=0.0002
embedding_dim = 512
text = ["Original Image","Image + Noise","Actual Noise","Predicted Noise"]
epoch = 100 
image_channels = 3 # input image channel
image_size = 256  # Reduced image size for this example
output_channel = 3 

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

ut = Utils()


# Train the Model
# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise


def train(epoch):
    generator,disc,gen_opt,disc_opt = ut.get_model(image_channels,output_channel,embedding_dim,
                                                              image_size,learning_rate,beta_value)

    dataloader = ut.gat_dataloader(base_dir=base_dir,data_filepath=xls_datapath,target_size=(image_size,image_size))    

    for e in range(epoch):

        generator.train()
        disc.train()
        gen_opt.param_groups[0]['lr'] = learning_rate*(1-e/epoch)
        disc_opt.param_groups[0]['lr'] = learning_rate*(1-e/epoch)
        i = 0
        
        for (image, emb_value,txt) in tqdm(dataloader):
            gen_opt.zero_grad()
            image = image.to(device)
            emb_value = emb_value.to(device)

            # perturb data
            noise = torch.randn_like(image)
            t = torch.randint(1, timesteps + 1, (image.shape[0],)).to(device)
            x_pert = perturb_input(image, t, noise)
            pt = generator(x_pert,emb_value,t/timesteps)

            # Train Discriminator
            real_pred = disc(noise)
            fake_pred = disc(pt.detach())
            real_loss = bce_loss(real_pred,torch.ones_like(real_pred))
            fake_loss = bce_loss(fake_pred,torch.zeros_like(fake_pred))
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            disc_opt.step()

            # Train Generator
            gfake_pred = disc(pt)
            gfake_loss = bce_loss(gfake_pred,torch.ones_like(gfake_pred))
            mloss = gfake_loss + mse_loss(pt,noise)
            mloss.backward()
            gen_opt.step()

            x_tack = torch.stack([image.squeeze().detach().cpu(),
                              x_pert.squeeze().detach().cpu(),
                              noise.squeeze().detach().cpu(),
                              pt.squeeze().detach().cpu()])
            print(f"Epoch: {e}, GAN Loss : {mloss.item()} , Disc Loss: {disc_loss.item()}")# Decoder Loss: {deloss.item()}")
            ut.show_tensor_images_train(x_tack,text)

            torch.save(obj=generator.state_dict(),f="load_model_weight/DGenerator.pth")
            torch.save(obj=disc.state_dict(),f='load_model_weight/DDisc.pth')

            if i%10 == 0 and i != 0:
                print(f"Clear Output {i}")
                clear_output()
            i += 1

            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Train Diffusion Generative Model
    train(epoch)