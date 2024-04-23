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
TRAIN_IMG_DIR = "/home/brian/Desktop/AIR_CV_Project-main/data/train_images/"
TRAIN_MASK_DIR = "/home/brian/Desktop/AIR_CV_Project-main/data/train_masks/"
VAL_IMG_DIR = "/home/brian/Desktop/AIR_CV_Project-main/data/val_images/"
VAL_MASK_DIR = "/home/brian/Desktop/AIR_CV_Project-main/data/val_masks/"

def main():
    model = UNET(in_channels=3, out_channels=1).to("cuda")
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    train_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

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
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()