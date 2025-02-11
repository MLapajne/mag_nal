# Train/Fine Tune SAM 2 on LabPics 1 dataset
# This mode use several images in a single batch
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

import numpy as np
import torch
import cv2
import os
import glob

from torch.onnx.symbolic_opset11 import hstack
from torch.utils.data import Dataset, DataLoader
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import scipy.ndimage as ndimage
from PIL import Image
from torchvision import transforms


###############################################################################
# Dataset definition
###############################################################################
class SAMDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(1024, 1024)):
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

        # For masks, we resize using nearest-neighbor interpolation to preserve label values.
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=Image.NEAREST),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
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


# Read data
"""
data_dir = r"LabPicsV1//"  # Path to dataset (LabPics 1)
data = []  # list of files in dataset
for ff, name in enumerate(
    os.listdir(data_dir + "Simple/Train/Image/")
):  # go over all folder annotation
    data.append(
        {
            "image": data_dir + "Simple/Train/Image/" + name,
            "annotation": data_dir + "Simple/Train/Instance/" + name[:-4] + ".png",
        }
    )
"""


def read_single(data):  # read random image and single mask from  the dataset (LabPics)

    #  select image

    ent = data[np.random.randint(len(data))]  # choose random entry
    Img = cv2.imread(ent["image"])[..., ::-1]  # read image
    ann_map = cv2.imread(ent["annotation"])  # read annotation

    # resize image

    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scalling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(
        ann_map,
        (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
        interpolation=cv2.INTER_NEAREST,
    )
    if Img.shape[0] < 1024:
        Img = np.concatenate(
            [Img, np.zeros([1024 - Img.shape[0], Img.shape[1], 3], dtype=np.uint8)],
            axis=0,
        )
        ann_map = np.concatenate(
            [
                ann_map,
                np.zeros(
                    [1024 - ann_map.shape[0], ann_map.shape[1], 3], dtype=np.uint8
                ),
            ],
            axis=0,
        )
    if Img.shape[1] < 1024:
        Img = np.concatenate(
            [Img, np.zeros([Img.shape[0], 1024 - Img.shape[1], 3], dtype=np.uint8)],
            axis=1,
        )
        ann_map = np.concatenate(
            [
                ann_map,
                np.zeros(
                    [ann_map.shape[0], 1024 - ann_map.shape[1], 3], dtype=np.uint8
                ),
            ],
            axis=1,
        )

    # merge vessels and materials annotations

    mat_map = ann_map[:, :, 0]  # material annotation map
    ves_map = ann_map[:, :, 2]  # vessel  annotaion map
    mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)  # merge maps

    # Get binary masks and points

    inds = np.unique(mat_map)[1:]  # load all indices
    if inds.__len__() > 0:
        ind = inds[np.random.randint(inds.__len__())]  # pick single segment
    else:
        return read_single(data)

    # for ind in inds:
    mask = (mat_map == ind).astype(
        np.uint8
    )  # make binary mask corresponding to index ind
    coords = np.argwhere(mask > 0)  # get all coordinates in mask
    yx = np.array(
        coords[np.random.randint(len(coords))]
    )  # choose random point/coordinate
    return Img, mask, [[yx[1], yx[0]]]


def read_batch(data, batch_size=4):
    limage = []
    lmask = []
    linput_point = []
    for i in range(batch_size):
        image, mask, input_point = read_single(data)
        limage.append(image)
        lmask.append(mask)
        linput_point.append(input_point)

    return limage, np.array(lmask), np.array(linput_point), np.ones([batch_size, 1])


def get_initial_point(mask_np):
    """
    Given a binary mask (numpy array, shape [H, W]) with 1 for the object,
    compute the point that is farthest from the boundary.
    We use a distance transform so that interior pixels get higher values.

    Returns:
        A tuple (x, y) in image coordinates.
    """
    # Compute distance transform on the foreground.
    # Note: distance_transform_edt computes distance for non-zero pixels.
    dt = ndimage.distance_transform_edt(mask_np)
    max_idx = np.unravel_index(np.argmax(dt), dt.shape)  # (row, col)
    # SAM expects (x, y) so we swap: col->x, row->y.
    return (int(max_idx[1]), int(max_idx[0]))


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sam2_checkpoint = "sam2.1_hiera_small.pt"  # path to model weight
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"  #  model config
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")  # load model
predictor = SAM2ImagePredictor(sam2_model)

batch_size = 2
image_size = (1024, 1024)
# Set training parameters

predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder
predictor.model.image_encoder.train(
    True
)  # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
optimizer = torch.optim.AdamW(
    params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5
)
scaler = torch.cuda.amp.GradScaler()  # mixed precision


# Create the dataset and dataloader.
train_dataset = SAMDataset(
    images_dir="split_images/split_1", masks_dir="mask_folder", image_size=image_size
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_loader:
        with torch.cuda.amp.autocast():  # cast to mix precision
            images, masks = batch  # images: [B, 3, H, W]; masks: [B, 1, H, W]
            images = images.to(device)
            masks = masks.to(device)
            predictor.set_image_batch(images)
            if masks.shape[0] == 0:
                continue  # ignore empty batches

            B = images.shape[0]
            for itr in range(B):
                gt_mask = masks[itr : itr + 1]
                gt_mask_np = gt_mask.squeeze().cpu().numpy()
                # apply SAM image encoder to the image
                # predictor.get_image_embedding()
                # prompt encoding

                input_label = np.ones([batch_size, 1])
                init_point = get_initial_point(gt_mask_np)
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    init_point,
                    input_label,
                    box=None,
                    mask_logits=None,
                    normalize_coords=True,
                )
                sparse_embeddings, dense_embeddings = (
                    predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, labels),
                        boxes=None,
                        masks=None,
                    )
                )

                # mask decoder

                high_res_features = [
                    feat_level[-1].unsqueeze(0)
                    for feat_level in predictor._features["high_res_feats"]
                ]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"],
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
                prd_masks = predictor._transforms.postprocess_masks(
                    low_res_masks, predictor._orig_hw[-1]
                )  # Upscale the masks to the original image resolution

                # Segmentaion Loss caclulation

                gt_mask = torch.tensor(gt_mask.astype(np.float32)).cuda()
                prd_mask = torch.sigmoid(
                    prd_masks[:, 0]
                )  # Turn logit map to probability map
                seg_loss = (
                    -gt_mask * torch.log(prd_mask + 0.00001)
                    - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)
                ).mean()  # cross entropy loss

                # Score loss calculation (intersection over union) IOU

                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                iou = inter / (
                    gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
                )
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                loss = seg_loss + score_loss * 0.05  # mix losses

                # apply back propogation

                predictor.model.zero_grad()  # empty gradient
                scaler.scale(loss).backward()  # Backpropogate
                scaler.step(optimizer)
                scaler.update()  # Mix precision

                if itr % 1000 == 0:
                    torch.save(
                        predictor.model.state_dict(), "model.torch"
                    )  # save model

                # Display results

                if itr == 0:
                    mean_iou = 0
                mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
                print("step)", itr, "Accuracy(IOU)=", mean_iou)
