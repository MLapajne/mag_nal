import numpy as np
from datasets import Dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from transformers import SamProcessor, SamModel
from torch.optim import Adam
import monai
import torch
from tqdm import tqdm
from statistics import mean
import glob
import cv2
from patchify import patchify  # Only to handle large images


to_tensor = ToTensor()
target_size = (1024, 1024)

# Get sorted list of image file paths
image_paths = sorted(
    glob.glob("/shared/home/matevz.lapajne/mag_nal/data/train1/images/*")
)
mask_paths = sorted(
    glob.glob("/shared/home/matevz.lapajne/mag_nal/data/train1/masks/*")
)

# Load and resize images
images = []
for p in image_paths:
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    images.append(img_resized)

masks = []
for p in mask_paths:
    mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    masks.append(mask_resized)

# Convert lists to numpy arrays
images_stack = np.stack(images, axis=0)
masks_stack = np.stack(masks, axis=0)
# Desired patch size for smaller images and step size.
patch_size = 256
step = 256

all_img_patches = []

for idx in range(images_stack.shape[0]):
    large_image = images_stack[idx]  # Color image with shape (1024, 1024, 3)
    # Define patch shape for a color image
    patch_shape = (patch_size, patch_size, large_image.shape[2])
    patches_img = patchify(large_image, patch_shape, step=step)

    # patches_img will have shape (n_patches_y, n_patches_x, patch_size, patch_size, channels)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, 0, :, :, :]
            all_img_patches.append(single_patch_img)

images = np.array(all_img_patches)


# Let us do the same for masks
all_mask_patches = []
for img in range(masks_stack.shape[0]):
    large_mask = masks_stack[img]
    patches_mask = patchify(large_mask, (patch_size, patch_size), step=step)
    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i, j, :, :]
            # print(np.unique(single_patch_mask))
            # single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
            all_mask_patches.append(single_patch_mask)
masks = np.array(all_mask_patches)

valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
filtered_images = images[valid_indices]
filtered_masks = masks[valid_indices]
# Convert the NumPy arrays to Pillow images and store them in a dictionary
dataset_dict = {
    "image": [Image.fromarray(img) for img in filtered_images],
    "label": [Image.fromarray(mask) for mask in filtered_masks],
}

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)


# Define the custom dataset class for SAM fine-tuning
class SAMDataset(Dataset):
    """
    This class creates a dataset that returns input images along with segmentation masks
    and prompt points derived from the masks.
    """

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = item["label"]

        input_points = self.get_points(ground_truth_mask)
        input_labels = [1, 0]

        # Prepare the inputs via the processor, now including the prompt points.
        inputs = self.processor(
            images=image,
            input_masks=ground_truth_mask,
            input_points=[input_points],
            input_labels=[input_labels],
            return_tensors="pt",
        )

        # Remove the batch dimension added by the processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add the ground truth mask for computing the loss later
        inputs["ground_truth_mask"] = to_tensor(ground_truth_mask)
        inputs["input_masks"] = to_tensor(ground_truth_mask)
        return inputs

    def get_points(ground_truth_mask):
        mask_array = np.array(ground_truth_mask)
        # Sample a foreground point (where mask > 0)
        foreground_coords = np.argwhere(mask_array > 0)
        if len(foreground_coords) > 0:
            fg_idx = np.random.choice(len(foreground_coords))
            fg_point = foreground_coords[fg_idx]
        else:
            # Fallback to center if no foreground exists
            fg_point = np.array([mask_array.shape[0] // 2, mask_array.shape[1] // 2])

        # Sample a background point (where mask == 0)
        background_coords = np.argwhere(mask_array == 0)
        if len(background_coords) > 0:
            bg_idx = np.random.choice(len(background_coords))
            bg_point = background_coords[bg_idx]
        else:
            bg_point = np.array([mask_array.shape[0] // 2, mask_array.shape[1] // 2])

        # Note: np.argwhere returns coordinates as (row, col); convert these to (x, y)
        fg_point = [int(fg_point[1]), int(fg_point[0])]
        bg_point = [int(bg_point[1]), int(bg_point[0])]

        # Prepare points and corresponding labels: 1 for foreground, 0 for background.
        # Wrap them in a list since the processor expects a list per image.
        input_points = [fg_point, bg_point]
        return input_points


processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
train_dataset = SAMDataset(dataset=dataset, processor=processor)

# Create a DataLoader instance for the training dataset
train_dataloader = DataLoader(
    train_dataset, batch_size=2, shuffle=True, drop_last=False
)


model = SamModel.from_pretrained("facebook/sam-vit-huge")

# Freeze vision encoder parameters so that only the mask decoder gets fine-tuned.
# for name, param in model.named_parameters():
#    if name.startswith("vision_encoder"):
#        param.requires_grad_(False)


# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")


# Training loop
num_epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
        # Move inputs to device and perform forward pass.
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            input_points=batch["input_points"].to(device),
            input_labels=batch["input_labels"].to(device),
            input_masks=batch["input_masks"].to(device),
            multimask_output=False,
        )

        # Compute loss using the predicted masks and ground truth.
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["input_masks"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks)

        # Backward pass and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f"EPOCH: {epoch}")
    print(f"Mean loss: {mean(epoch_losses)}")


torch.save(
    model.state_dict(), "/shared/home/matevz.lapajne/mag_nal/mito_model_checkpoint.pth"
)
