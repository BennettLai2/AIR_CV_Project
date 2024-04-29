import os
import cv2
import numpy as np
from patchify import patchify
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from model import UNET
from utils import (load_checkpoint)
from PIL import Image
from matplotlib import pyplot as plt

# Change here. Put your input image in user/input. Make sure there is only 1 file. 
input_path = "user/input"
temp_path = "user/temp"
output_path = "user/output"
image_name = "img_1.jpg"
ckpt = "my_checkpoint_advanced.pth.tar"
overlay_color = (255, 0, 0)
#################

def rename_files(input_folder):
    # List all files in the directory
    files = os.listdir(input_folder)

    for file in files:
        # Check if the filename starts with 'mask_'
        if file.startswith("mask_"):
            # Generate the new filename by removing 'mask_'
            new_filename = file.replace("mask_", "")
            # Create full paths
            original_path = os.path.join(input_folder, file)
            new_path = os.path.join(input_folder, new_filename)
            # Rename the file
            os.rename(original_path, new_path)
            print(f"Renamed '{file}' to '{new_filename}'")

# if __name__ == "__main__":
#     # Specify the folder containing the mask files
#     input_folder = "./temp/train_masks"  # Change this to your actual folder path
#     rename_files(input_folder)


original_height = None
original_width = None

def create_mask(input_folder, output_folder):
    global original_height, original_width
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of image filenames in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to open image file {image_path}")
            continue
            
#         original_height, original_width = image.shape[:2]
        if original_height is None or original_width is None:
                    original_height, original_width = image.shape[:2]

        # Resize the image to make dimensions divisible by 256
        height, width = image.shape[:2]
        new_height = height - (height % 256)
        new_width = width - (width % 256)

        if new_height == 0 or new_width == 0:
            print(f"Image {image_path} is too small for resizing.")
            continue

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create patches
        patches_img = patchify(image, (256, 256, 3), step=256)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, 0]
                # Save the patch
                mask_filename = os.path.join(output_folder, f"{image_file[:-4]}_{i}_{j}.jpg")
                cv2.imwrite(mask_filename, single_patch_img)

    print("Masks created and saved.")

# if __name__ == "__main__":
#     # Input folder containing original images
#     images_input = "user/input"
    
#     # Output folder for images
#     images_output = "user/temp"

#     # Create masks
#     create_mask(images_input, images_output)

# Step 1: Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to match model input size
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            ),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    # print(image.shape)
    return image

# Step 2: Load your pre-trained PyTorch model
model = UNET(in_channels=3, out_channels=1).to("cuda")
load_checkpoint(torch.load(ckpt), model)
model.eval()  # Set the model to evaluation mode

# Step 3: Make prediction on the image
def predict_image(image_path, model):
    # Preprocess the image
    input_image = preprocess_image(image_path)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_image.to(device="cuda"))
        output = torch.sigmoid(output)
        output = torch.where(output > .1, torch.tensor(1.0), torch.tensor(0.0))
    
    # Interpret the output (assuming output is a mask)
    predicted_mask = output.squeeze().cpu().numpy()  # Convert tensor to numpy array and remove batch dimension
    
    return predicted_mask

# Step 4: Use the function to make a prediction
def pred_images(image_path): 
    predicted_mask = predict_image(image_path, model)
    # np.set_printoptions(threshold=np.inf)
    # print(predicted_mask)

    # Step 5: Visualize the predicted mask
    # plt.imshow(predicted_mask)
    # plt.axis('off')
    # plt.show()
    image = Image.fromarray((predicted_mask*255).astype(np.uint8))
    image.save(image_path)

def reconstruct_image(patches_folder, reconstructed_folder):
    global original_height, original_width
    # Create the reconstructed folder if it doesn't exist
    if not os.path.exists(reconstructed_folder):
        os.makedirs(reconstructed_folder)

    files = os.listdir(patches_folder)
    unique_ids = set('_'.join(file.split('_')[:2]) for file in files if file.endswith(('.jpg', '.jpeg', '.png')))

    for image_id in unique_ids:
        patch_files = [f for f in files if f.startswith(image_id) and f.endswith(('.jpg', '.jpeg', '.png'))]
        patch_files.sort()

        # Determine dimensions for reconstruction
        max_i = max_j = 0
        for file in patch_files:
            parts = file.split('_')
            i, j = int(parts[2]), int(parts[3].split('.')[0])
            max_i = max(max_i, i)
            max_j = max(max_j, j)

        patch_size = 256
        height = (max_i + 1) * patch_size
        width = (max_j + 1) * patch_size
        reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Place each patch in the correct position
        for file in patch_files:
            parts = file.split('_')
            i, j = int(parts[2]), int(parts[3].split('.')[0])
            patch = cv2.imread(os.path.join(patches_folder, file))
            reconstructed_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patch

        # Save the reconstructed image
        reconstructed_filename = os.path.join(reconstructed_folder, f'{image_id}.jpg')
        cv2.imwrite(reconstructed_filename, reconstructed_image)
        
        # Resize to the original dimensions
        resized_image = cv2.resize(reconstructed_image, (original_width, original_height), interpolation=cv2.INTER_AREA)
        reconstructed_filename = os.path.join(reconstructed_folder, f'{image_id}.jpg')
        cv2.imwrite(reconstructed_filename, resized_image)

    print("Images reconstructed and saved.")

# if __name__ == "__main__":
    
#     # IMAGES ######################################################
#     # Paths for input and output
#     input_folder = "user/input"
#     output_folder = "user/temp"
#     reconstructed_folder = "user/output"

#     # Reconstruct images
#     reconstruct_image(output_folder, reconstructed_folder)


def overlay(img_path, mask_path): 
    full_image = Image.open(img_path)
    full_pixels = full_image.load()

    binary_image = Image.open(mask_path).convert('L')
    # plt.imshow(binary_image)
    # plt.axis('off')
    # plt.show()
    binary_pixels = binary_image.load()

    # Define the RGB value for pink
    color = overlay_color  # Adjust the values as needed

    # Iterate over each pixel of the binary image
    width, height = binary_image.size
    for x in range(width):
        for y in range(height):
            # Check if the pixel value in the binary image is 1
            if binary_pixels[x, y] > 150:
                # Set the corresponding pixel in the full image to color
                # print(binary_pixels[x, y])
                full_pixels[x, y] = color

    # Save the modified full image
    full_image.save(mask_path)  # Save the modified image

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Deletes the file
            # If you want to delete subdirectories recursively, you can use os.rmdir() instead
            # else:
            #     os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

delete_files_in_folder(temp_path)
delete_files_in_folder(output_path)

create_mask(input_path, temp_path)
# Get a list of image filenames in the input folder
image_files = [f for f in os.listdir(temp_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    pred_images(temp_path+"/"+str(image_file))

# Reconstruct images
reconstruct_image(temp_path, output_path)

overlay(input_path + "/" + image_name, output_path + "/" + image_name)
