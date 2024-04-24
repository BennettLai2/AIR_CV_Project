# AIR_CV_Project

Group members: Bennett Lai, Brian Lee, Luis Martinez Morales
This our the CV project for AI Robotics with Dr. Robin Murphy.

Working Environment:
Linux OS using Ubuntu 20.0.4
Pytorch version 2.1.2
CUDA version PyTorch built with 12.1

User component: 
Place your input image in user/input. Make sure it is the only image in there. 
Go to User_Testing.ipynb. 
Run all the blocks in User_Testing.ipynb. Modify the last block. 
See user/output for result

Data:
Data/train_images, train_masks, val_images, val_masks
These are processed images, with their masks

Data/p_train_images, ...
These are the original images, with their masks

Pre_training:
First, we got 3 videos of the river. The footages added up to be ~15 minuts long.
Use extract_frames() in imgPulling.ipynb to extract 1 frame per second, and save it into a temp folder we named "img".

Then we used pick_and_zip_images() to zip the folders, and sent them out for annotation.
We saved the annotated images in the folder "img_annotated".

We used the #FF7F27 color for annotation, so we can extract that color from the annotated images to create a mask.
Use create_mask() in img.Mask.ipynb to extract the masks from "img_annotated".
The created masks were saved in "img_mask".

Then, we cut the img and masks into 256\*256 images, and saved them into data/train_images train_masks val_images and val_masks.
Use create_patch() in imgPatchify.ipynb to cut the images. If your image isn't divisible by 256, resize the image so that it is.
Since there were too many images with no debris, we used the second cell in imgPatchify.ipynb to drop 80% of the uninteresing images.

Training:
Training is rather simple. Once you have all the images processed into 256\*256 rgb blocks, go to train.py.
In there, make sure all the hyper parameters are good. If you have an existing model you want to improve on, set LOAD_MODEL = True.
Check that the directories are correct, and that the image width and height are good.

Then run:
python train.py

And the training should start. You will see the loss and epoch infomation there.
Once it is done, you can see some preview images in "saved_images"
and the model should be saved via save_checkpoint().

Test and Eval:
Our testing returns the Accuracy, Dice Score, and a modified binary-accuracy of our own design.
Accuracy returns the percentage of pixels where Mask == Pred.
Dice Score returns the percentage overlap for the set of pixels == 1.
The binary-accuracy returns the TP, TN, FP, FN of the predictions, but instead, treats each mask as 0/1.
i.e. if there is debris, set it to 1, else 0. We ignore the outer 10 pixels for this analysis. (left, right, top, bot 10 px).
because it is unreasonable to find a debris from that little information.(Humans can't do it)

For the validation set
Acc: 155373271/155975680 = 99.61
Dice: 0.521
Bin: 2278/2380 = 95.71
TP: 337
TN: 1941
FP: 39
FN: 63

run:
python test.py
