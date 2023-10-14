import torch
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

def mse_loss(y_pred,y_true):
    return F.mse_loss(y_pred,y_true)

