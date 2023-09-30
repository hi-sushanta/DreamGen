import torch
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a pretrained VGG16 model
vgg16 = models.vgg16(pretrained=True).to(device)
vgg16_features = vgg16.features

# Define a transform to preprocess images for the VGG model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to VGG input size
])

def perceptual_loss(input, target):
    # Preprocess input and target images
    input = preprocess(input)
    target = preprocess(target)

    # Calculate VGG feature maps
    input_features = vgg16_features(input)
    target_features = vgg16_features(target)

    # Compute feature-wise L1 loss
    loss = 0.0
    for input_feat, target_feat in zip(input_features, target_features):
        loss += F.l1_loss(input_feat, target_feat)

    return loss

def mse_loss(y_pred,y_true):
    return nn.MSELoss()(y_pred,y_true)

def bce_loss(y_pred,y_true):
    return nn.BCELoss()(y_pred,y_true)