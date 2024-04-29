import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (load_checkpoint, get_loaders, check_accuracy)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
BATCH_SIZE = 16
NUM_EPOCHS = 0
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = True
MODEL_NAME = "my_checkpoint.pth.tar"
TRAIN_IMG_DIR = "data/s_train_images/"
TRAIN_MASK_DIR = "data/s_train_masks/"
VAL_IMG_DIR = "data/s_val_images/"
VAL_MASK_DIR = "data/s_val_masks/"

def main():
    model = UNET(in_channels=3, out_channels=1).to("cuda")
    load_checkpoint(torch.load(MODEL_NAME), model)

    val_transforms = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        None,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()