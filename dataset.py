import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PolypDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Get lists of image and mask file names
        # self.image_names = sorted(os.listdir(images_dir))
        # self.mask_names = sorted(os.listdir(masks_dir))

        self.image_names = os.listdir(images_dir)
        self.mask_names = os.listdir(masks_dir)
        # Ensure both directories have the same number of files
        assert len(self.image_names) == len(self.mask_names), "Mismatch between images and masks count."

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = self.mask_names[idx]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Image file not found: {img_path}")
            return None  # Handle missing image appropriately

        # Load mask
        try:
            mask = Image.open(mask_path).convert('L')  # Assuming masks are single-channel
        except FileNotFoundError:
            print(f"Mask file not found: {mask_path}")
            return None  # Handle missing mask appropriately

        # Apply transformations
        """
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)  # Apply same transformations to masks if needed
        """
        if self.image_transform:
            img = self.image_transform(img)
            # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # img = normalize(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return img, mask