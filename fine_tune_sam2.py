#!/usr/bin/env python3
"""
Script to fine-tune the SAM model using interactive point prompts.
We simulate interactive segmentation:
  - The first prompt is the positive point farthest from the object boundary.
  - In subsequent rounds, we compute the error region (difference between the prediction
    and the ground-truth mask) and add a new prompt at the pixel farthest from the error boundary.
  
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

# Import the SAM model from the official repository.
from segment_anything import sam_model_registry

# SciPy’s ndimage is used for computing distance transforms.
import scipy.ndimage as ndimage

import argparse
import logging
from tqdm import tqdm
from typing import Optional, Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


###############################################################################
# Dataset definition
###############################################################################
class SAMDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: Tuple[int, int] = (1024, 1024),
    ) -> None:
        """
        Args:
            images_dir (str): Path to images.
            masks_dir (str): Path to binary masks.
            image_size (tuple): (width, height) to which images and masks will be resized.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        # Assumes images and masks have matching filenames.
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*")))
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Number of images and masks do not match.")

        self.image_size = image_size

        # Transform for images: resize and convert to tensor.
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # (Optionally) add normalization here if required by SAM’s image encoder.
            ]
        )

        # For masks, use nearest-neighbor to preserve labels.
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=Image.NEAREST),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and its corresponding mask.
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = np.array(mask)
        # Binarize mask (assumes 0 vs. 255; adjust threshold if needed).
        mask = (mask > 128).astype(np.float32)
        # Convert mask to tensor with shape [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


###############################################################################
# Helper functions for point sampling
###############################################################################
def get_initial_point(mask_np: np.ndarray) -> Tuple[int, int]:
    """
    Given a binary mask (numpy array, shape [H, W]) with 1 for the object,
    compute the point that is farthest from the boundary using a distance transform.

    Returns:
        A tuple (x, y) in image coordinates.
    """
    dt = ndimage.distance_transform_edt(mask_np)
    max_idx = np.unravel_index(np.argmax(dt), dt.shape)  # (row, col)
    # SAM expects (x, y) so we swap: col -> x, row -> y.
    return (int(max_idx[1]), int(max_idx[0]))


def get_interactive_point(
    gt_mask_np: np.ndarray, error_np: np.ndarray
) -> Optional[Tuple[int, int]]:
    """
    Given the ground truth mask and an error mask (both numpy arrays, shape [H, W]),
    select the error pixel that is farthest from the error boundary.

    Args:
        gt_mask_np (np.ndarray): Ground truth binary mask.
        error_np (np.ndarray): Binary error mask (1 where prediction != ground truth).

    Returns:
        A tuple (row, col) representing the pixel location in the error region,
        or None if no error is present.
    """
    if error_np.sum() == 0:
        return None
    dt = ndimage.distance_transform_edt(error_np)
    max_idx = np.unravel_index(np.argmax(dt), dt.shape)  # (row, col)
    return (int(max_idx[0]), int(max_idx[1]))


###############################################################################
# Interactive segmentation simulation function
###############################################################################
def simulate_interactive_segmentation(
    image_embedding: torch.Tensor,
    gt_mask: torch.Tensor,
    image_size: Tuple[int, int],
    sam,
    device: torch.device,
    max_interactions: int = 3,
) -> torch.Tensor:
    """
    Simulate interactive segmentation on a single image.

    Args:
        image_embedding (torch.Tensor): Pre-computed embedding for the image.
        gt_mask (torch.Tensor): Ground truth mask for the image.
        image_size (tuple): (width, height) of the image.
        sam: The SAM model.
        device (torch.device): Computation device.
        max_interactions (int): Maximum number of interactive prompts.

    Returns:
        final_pred_logits (torch.Tensor): The final prediction logits.
    """
    gt_mask_np = gt_mask.squeeze().cpu().numpy()  # shape: [H, W]
    points_list: List[Tuple[int, int]] = []
    labels_list: List[int] = []

    # 1. Initial point: choose the positive point farthest from the boundary.
    init_point = get_initial_point(gt_mask_np)  # returns (x, y)
    points_list.append(init_point)
    labels_list.append(1)

    final_pred_logits: Optional[torch.Tensor] = None

    for _ in range(max_interactions):
        # Prepare prompt tensors.
        prompt_points = torch.tensor(
            points_list, dtype=torch.float32, device=device
        ).unsqueeze(0)
        prompt_labels = torch.tensor(
            labels_list, dtype=torch.int, device=device
        ).unsqueeze(0)

        prompt_embeddings = sam.prompt_encoder(
            points=(prompt_points, prompt_labels), boxes=None, masks=None
        )

        masks_pred, _, _ = sam.mask_decoder(
            image_embeddings=image_embedding.unsqueeze(0),
            prompt_embeddings=prompt_embeddings,
            multimask_output=False,
        )

        masks_pred_upsampled = torch.nn.functional.interpolate(
            masks_pred,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        final_pred_logits = masks_pred_upsampled

        # Generate binary prediction.
        pred_mask = torch.sigmoid(masks_pred_upsampled)
        pred_binary = (pred_mask > 0.5).float()
        pred_binary_np = pred_binary.squeeze().cpu().numpy()  # shape: [H, W]

        # Compute error region.
        error_np = (pred_binary_np != gt_mask_np).astype(np.uint8)
        if error_np.sum() == 0:
            break

        new_point_rc = get_interactive_point(gt_mask_np, error_np)
        if new_point_rc is None:
            break

        # Convert (row, col) to (x, y) format expected by SAM.
        new_point_xy = (new_point_rc[1], new_point_rc[0])
        label = 1 if gt_mask_np[new_point_rc[0], new_point_rc[1]] == 1 else 0
        points_list.append(new_point_xy)
        labels_list.append(label)

    if final_pred_logits is None:
        raise RuntimeError(
            "No prediction was generated during interactive segmentation."
        )

    return final_pred_logits


###############################################################################
# Training function
###############################################################################
def train_sam(
    images_dir: str = "/kaggle/working/mag_nal/data/train1/images",
    masks_dir: str = "/kaggle/working/mag_nal/data/train1/masks",
    num_epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    image_width: int = 1024,
    image_height: int = 1024,
    max_interactions: int = 3,
    model_type: str = "vit_b",
    checkpoint_path: str = "/kaggle/working/mag_nal/models/sam/sam_vit_b_01ec64.pth",
    checkpoint_dir: str = "/kaggle/working/mag_nal/models/sam",
    num_workers: int = 4,
    freeze_encoder: bool = False,
    use_amp: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_dataset = SAMDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=(image_width, image_height),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.train()

    if freeze_encoder:
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        logger.info("Image encoder frozen.")

    optimizer = optim.Adam(sam.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    use_amp = device.type == "cuda" and use_amp
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    logger.info("Starting training ...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        sam.train()
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            image_embeddings = sam.image_encoder(images)
            batch_loss = 0.0
            B = images.shape[0]

            for i in range(B):
                gt_mask = masks[i : i + 1]
                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred_logits = simulate_interactive_segmentation(
                            image_embedding=image_embeddings[i],
                            gt_mask=gt_mask,
                            image_size=(image_width, image_height),
                            sam=sam,
                            device=device,
                            max_interactions=max_interactions,
                        )
                        loss = criterion(pred_logits, gt_mask)
                else:
                    pred_logits = simulate_interactive_segmentation(
                        image_embedding=image_embeddings[i],
                        gt_mask=gt_mask,
                        image_size=(image_width, image_height),
                        sam=sam,
                        device=device,
                        max_interactions=max_interactions,
                    )
                    loss = criterion(pred_logits, gt_mask)

                batch_loss += loss

            batch_loss = batch_loss / B
            if use_amp:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()

            epoch_loss += batch_loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"sam_finetuned_epoch{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": sam.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete.")


###############################################################################
# Main function
###############################################################################
def main():
    train_sam()


if __name__ == "__main__":
    main()
