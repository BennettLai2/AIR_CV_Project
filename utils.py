import torch
import torchvision
from dataset import RiverDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = RiverDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = RiverDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct_pix = 0
    num_pixels = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    total_images = 0
    dice_score_pix = 0
    dice_score_img = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Pixel Based Accuracy
            num_correct_pix += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score_pix += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

            # Image Based Accuracy
            crop_slice = slice(10,-10)
            cropped_preds = preds[:, :, crop_slice, crop_slice]
            cropped_y = y[:, :, crop_slice, crop_slice]
            preds_binary = cropped_preds.reshape(cropped_preds.size(0), -1).max(1)[0]
            y_binary = cropped_y.reshape(cropped_y.size(0), -1).max(1)[0]
            false_positives += ((preds_binary == 1) & (y_binary == 0)).sum().item()
            false_negatives += ((preds_binary == 0) & (y_binary == 1)).sum().item()
            true_positives += ((preds_binary == 1) & (y_binary == 1)).sum().item()
            true_negatives += ((preds_binary == 0) & (y_binary == 0)).sum().item()
            total_images += preds.size(0)

    print("Pixel Based Metrics:")
    print(
        f"Got {num_correct_pix}/{num_pixels} with acc {num_correct_pix/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score_pix/len(loader)}")
    print()
    print("Image Based Metrics:")
    num_correct_img = true_positives + true_negatives
    print(
        f"Got {num_correct_img}/{total_images} with acc {num_correct_img/total_images*100:.2f}"
    )
    print(
        f"Got {true_positives} true positives"
    )
    print(
        f"Got {true_negatives} true negatives"
    )
    print(
        f"Got {false_positives} false positives"
    )
    print(
        f"Got {false_negatives} false negatives"
    )
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        torchvision.utils.save_image(
            x, f"{folder}/actual_{idx}.png"
        )
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()