import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# getting the dataset
class RiverDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # the path of images and masks
        # the masks have the name mask_{original image name}.jpg
        # it's just how it's setup, please follow the format when generating new images
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, "mask_"+str(self.images[index]))

        # converting image to rgb, mask to grayscale, then converting mask to binary
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask =  np.where(mask > 10, 255.0, 0.0)
        mask[mask == 255.0] = 1.0

        # image augmentation 
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask