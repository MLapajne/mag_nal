#!/usr/bin/env python3
"""
Example script to fine-tune the SAM model on a custom binary segmentation dataset.

Dataset structure (example):
    data/
      images/
         image1.jpg
         image2.jpg
         ...
      masks/
         image1.png   # binary mask for image1.jpg
         image2.png
         ...

This script uses a “bounding box prompt” computed on the fly from the binary mask.
You may need to adjust parts of this script depending on your SAM code version.
"""

import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import SAM model from the segment-anything repository.
# (Make sure that you have the repository in your PYTHONPATH or installed as a package.)
from segment_anything import sam_model_registry  # provided by the official SAM repo


###############################################################################
# Custom Dataset
###############################################################################
class SAMDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(1024, 1024)):
        """
        Args:
            images_dir (str): Path to images.
            masks_dir (str): Path to binary masks.
            image_size (tuple): Desired (width, height) to resize images and masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        # List files (assumes images and masks have matching filenames)
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*")))
        assert len(self.image_paths) == len(
            self.mask_paths
        ), "Mismatch between number of images and masks"

        self.image_size = image_size
        # Define a basic transform for the image (convert to tensor, normalize to [0,1])
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # (Optional) add normalization here if required by SAM’s image encoder
            ]
        )
        # For the mask, we just resize and convert to tensor
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=Image.NEAREST),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        # Resize (using our transforms)
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = np.array(mask)
        # Binarize mask: convert to 0 and 1 (adjust threshold if needed)
        mask = (mask > 128).astype(np.float32)

        # Compute bounding box coordinates from the mask.
        # (Coordinates are in (x_min, y_min, x_max, y_max) format in pixel units.)
        coords = np.argwhere(mask > 0)
        if coords.size > 0:
            # Note: np.argwhere returns (row, col) == (y, x)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
        else:
            # If no foreground, use the full image.
            x_min, y_min = 0, 0
            x_max, y_max = self.image_size[0] - 1, self.image_size[1] - 1

        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        # Convert mask to a tensor with shape [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Convert bounding box to a tensor (shape [4])
        box = torch.from_numpy(box)

        return image, mask, box


###############################################################################
# Training Function
###############################################################################
def train_sam():
    # Hyper-parameters (adjust as needed)
    num_epochs = 10
    batch_size = 4
    learning_rate = 1e-4
    image_size = (1024, 1024)  # (width, height) expected by SAM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the dataset and dataloader (adjust the paths if necessary)
    train_dataset = SAMDataset(
        images_dir="data/images", masks_dir="data/masks", image_size=image_size
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Load the SAM model.
    # Choose the model type ("vit_b" or "vit_l" etc.) and provide the pretrained checkpoint path.
    model_type = "vit_b"
    checkpoint_path = "sam_vit_b_01ec64.pth"  # update this path to where you downloaded the checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.train()  # set to training mode

    # Optionally freeze the image encoder (which is very heavy) so that only the prompt and mask decoder are updated:
    # for param in sam.image_encoder.parameters():
    #     param.requires_grad = False

    # Define the optimizer (here, Adam) and loss function.
    optimizer = optim.Adam(sam.parameters(), lr=learning_rate)
    criterion = (
        nn.BCEWithLogitsLoss()
    )  # SAM’s mask decoder outputs logits; compare to binary ground truth

    print("Starting training ...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (images, masks, boxes) in enumerate(train_loader):
            images = images.to(device)  # [B, 3, H, W]
            masks = masks.to(device)  # [B, 1, H, W]
            boxes = boxes.to(device)  # [B, 4]

            optimizer.zero_grad()

            # === Forward pass through SAM ===
            # 1. Encode the image
            image_embeddings = sam.image_encoder(images)

            # 2. Prepare the prompt.
            # SAM’s prompt encoder can take boxes (and/or points). Here we pass the bounding boxes.
            prompt_embeddings = sam.prompt_encoder(boxes=boxes)

            # 3. Decode the mask.
            # The mask decoder returns a mask prediction (and additional outputs, e.g. predicted IoU scores).
            # We use multimask_output=False to return a single mask per prompt.
            masks_pred, iou_preds, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                prompt_embeddings=prompt_embeddings,
                multimask_output=False,
            )
            # masks_pred shape: [B, 1, H_out, W_out]
            # Usually H_out and W_out are lower than the original input size.
            # Upsample the predicted masks to match the ground truth resolution.
            masks_pred = torch.nn.functional.interpolate(
                masks_pred, size=image_size, mode="bilinear", align_corners=False
            )

            # === Loss computation ===
            loss = criterion(masks_pred, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

        # Optionally, save a checkpoint after each epoch.
        ckpt_path = f"sam_finetuned_epoch{epoch+1}.pth"
        torch.save(sam.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    train_sam()
