import torch
import torch.nn.functional as F
from pathlib import Path

from backend.processing.model import UVDocnet

IMG_SIZE = [488, 712]

CKPT_PATH = Path("app/models/preprocessing/best_model.pkl")

def load_model():
    """
    Charge UVDocnet depuis un fichier .pkl (checkpoint).
    Gestion automatique des formats model_state / state_dict.
    """
    # model = UVDocnet(num_filter=32, kernel_size=5)
    # ckpt = torch.load(ckpt_path)
    # model.load_state_dict(ckpt["model_state"])
    # return model

    ckpt = torch.load(CKPT_PATH, map_location=torch.device("cpu"))
    model = UVDocnet(num_filter=32, kernel_size=5)

    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError(f"Impossible de trouver 'model_state' ou 'state_dict' dans {ckpt_path}")

    # Charge dans le modèle
    model.load_state_dict(state_dict)

    return model

def bilinear_unwarping(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Args:
        warped_img  :       torch.Tensor of shape BxCxHxW (dtype float)
        point_positions:    torch.Tensor of shape Bx2xGhxGw (dtype float)
        img_size:           tuple of int [w, h]
    """
    upsampled_grid = F.interpolate(
        point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True)

    return unwarped_img