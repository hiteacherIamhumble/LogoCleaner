import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import Dict, Tuple, Optional, List
import random

class LogoDetectionDataset(Dataset):
    """
    Dataset for logo detection with image and corresponding mask pairs,
    specifically designed for use with SlimSAM and training a Selection Head.
    """
    def __init__(self, 
                 root_dir: str, 
                 filelist_path: str, 
                 image_size: int = 1024,
                 sam_image_size: Optional[int] = None,
                 transform=None, 
                 mask_transform=None,
                 augment: bool = True):
        """
        Args:
            root_dir (string): Root directory of the dataset
            filelist_path (string): Path to filelist-logosonly.txt
            image_size (int): Size to resize images for the Selection Head
            sam_image_size (int, optional): Size to resize images for SlimSAM input (if different)
            transform (callable, optional): Optional transform for images
            mask_transform (callable, optional): Optional transform for masks
            augment (bool): Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_size = image_size
        self.sam_image_size = sam_image_size if sam_image_size else image_size
        self.augment = augment
        
        # Read filelist
        with open(filelist_path, 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines()]
        print(f"Found {len(self.file_paths)} images in {filelist_path}")
        
        # Store image paths and mask counts
        self.data_pairs = []
        for img_path in self.file_paths:
            # Get full path
            full_img_path = os.path.join('/mnt1/peter/datasets/train/', img_path.lstrip('./'))
            
            # Get directory and filename
            img_dir, img_filename = os.path.split(full_img_path)
            img_name = os.path.splitext(img_filename)[0]
            
            # Read gt_data.txt to get mask count
            gt_data_path = os.path.join(img_dir, f"{img_name}.gt_data.txt")
            if os.path.exists(gt_data_path):
                
                with open(gt_data_path, 'r') as gt_file:
                    mask_count = len(gt_file.readlines())
            else:
                # If gt_data.txt doesn't exist, try to determine mask count by finding files
                mask_count = 0
                while os.path.exists(os.path.join(img_dir, f"{img_name}.mask{mask_count:02d}.png")):
                    mask_count += 1
            
            # Store image path and mask count
            for mask_idx in range(mask_count):
                mask_path = os.path.join(img_dir, f"{img_name}.mask{mask_idx:02d}.png")
                if os.path.exists(mask_path):
                    self.data_pairs.append({
                        'image_path': full_img_path,
                        'mask_path': mask_path
                    })

    def __len__(self):
        return len(self.data_pairs)
    
    def _apply_shared_transforms(self, image, mask):
        """Apply shared transforms to both image and mask to ensure alignment."""
        # Convert to PIL images if they're tensors
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        if isinstance(mask, torch.Tensor):
            mask = TF.to_pil_image(mask)
            
        # Get original dimensions
        orig_w, orig_h = image.size
        
        # Resize keeping aspect ratio
        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size), 
                        interpolation=transforms.InterpolationMode.NEAREST)
        
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
            # Random rotation
            if random.random() > 0.5:
                angle = random.choice([-30, -15, 0, 15, 30])
                image = TF.rotate(image, angle, fill=0)
                mask = TF.rotate(mask, angle, fill=0)
                
            # Random color jitter (only for image)
            if random.random() > 0.5:
                brightness = 0.1 + random.random() * 0.2  # 0.1-0.3
                contrast = 0.1 + random.random() * 0.2    # 0.1-0.3
                saturation = 0.1 + random.random() * 0.2  # 0.1-0.3
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)
                image = TF.adjust_saturation(image, saturation)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Normalize image for SlimSAM (typically ImageNet normalization)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image and mask paths
        img_path = self.data_pairs[idx]['image_path']
        mask_path = self.data_pairs[idx]['mask_path']
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load as grayscale
        
        # Get original dimensions for potential reference
        orig_w, orig_h = image.size
        
        # Store original image for SlimSAM if different size is required
        if self.sam_image_size != self.image_size:
            # Create a separate copy for SAM input if needed
            sam_image = TF.resize(image, (self.sam_image_size, self.sam_image_size))
            sam_image = TF.to_tensor(sam_image)
            sam_image = TF.normalize(sam_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Apply shared transforms that keep image and mask aligned
        image, mask = self._apply_shared_transforms(image, mask)
        
        # Create additional metadata useful for training with SAM
        # Calculate bounding box from mask
        if mask.sum() > 0:  # Ensure mask isn't empty
            # Find bounding box coordinates from mask
            y_indices, x_indices = torch.where(mask[0] > 0)
            x_min, x_max = x_indices.min().item(), x_indices.max().item()
            y_min, y_max = y_indices.min().item(), y_indices.max().item()
            bbox = torch.tensor([x_min, y_min, x_max, y_max])
            
            # Calculate box center point and scale for point-prompting
            center_point = torch.tensor([
                (x_min + x_max) / 2 / self.image_size, 
                (y_min + y_max) / 2 / self.image_size
            ])
        else:
            # Fallback for empty masks
            bbox = torch.tensor([0, 0, 1, 1])
            center_point = torch.tensor([0.5, 0.5])
            
        output = {
            'image': image,  # For main model input
            'mask': mask,    # Ground truth mask
            'bbox': bbox,    # Bounding box of the logo
            'point_coords': center_point.unsqueeze(0),  # Center point as SAM prompt
            'point_labels': torch.ones(1),  # Foreground point
            'original_size': torch.tensor([orig_h, orig_w]),  # Original image dimensions
        }
        
        # Add SAM-specific image if using different resolution
        if self.sam_image_size != self.image_size:
            output['sam_image'] = sam_image
            
        return output


def create_logo_dataloader(
    root_dir='./datasets', 
    filelist_path='./datasets/train/filelist-logosonly.txt',
    batch_size=8,
    image_size=1024,  # Default size for Selection Head
    sam_image_size=None,  # Optional different size for SlimSAM
    num_workers=4,
    shuffle=False,
    augment=False
):
    """
    Create a dataloader for the logo detection dataset optimized for SlimSAM training.
    
    Args:
        root_dir (string): Root directory of the dataset
        filelist_path (string): Path to filelist-logosonly.txt
        batch_size (int): Batch size for dataloader
        image_size (int): Size to resize images for Selection Head
        sam_image_size (int, optional): Size to resize images for SlimSAM input
        num_workers (int): Number of worker threads for dataloader
        shuffle (bool): Whether to shuffle the dataset
        augment (bool): Whether to apply data augmentation
        
    Returns:
        dataloader (DataLoader): PyTorch DataLoader for the dataset
    """
    # Create dataset
    dataset = LogoDetectionDataset(
        root_dir=root_dir,
        filelist_path=filelist_path,
        image_size=image_size,
        sam_image_size=sam_image_size,
        augment=augment
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    # Create dataloader
    dataloader = create_logo_dataloader(
        root_dir='./datasets',
        filelist_path='./datasets/train/filelist-logosonly.txt',
        batch_size=8
    )
    
    # Test dataloader
    for batch in dataloader:
        images = batch['image']
        masks = batch['mask']
        print(f"Batch images shape: {images.shape}")
        print(f"Batch masks shape: {masks.shape}")
        # Process just one batch for testing
        break
        
    print(f"Dataset contains {len(dataloader.dataset)} image-mask pairs.")
    print(f"Number of batches: {len(dataloader)}")