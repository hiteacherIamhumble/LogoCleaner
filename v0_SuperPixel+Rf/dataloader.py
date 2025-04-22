import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict

class LogoDatasetCombined(Dataset):
    def __init__(self, root_dir, filelist_path, image_size=1024, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        
        # Read filelist
        with open(filelist_path, 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines()]
        print(f"Found {len(self.file_paths)} images in {filelist_path}")
        
        # Group images and masks
        self.image_data = []
        valid_images = 0
        skipped_images = 0
        
        # First pass: group masks by image
        for img_path in self.file_paths:
            try:
                # Get full path
                if 'train' in filelist_path:
                    full_img_path = os.path.join('./datasets/train/', img_path.lstrip('./'))
                else:
                    full_img_path = os.path.join('./datasets/test/', img_path.lstrip('./'))
                
                # Verify image exists
                if not os.path.exists(full_img_path):
                    print(f"Warning: Image not found: {full_img_path}")
                    skipped_images += 1
                    continue
                
                # Try opening the image to verify it's valid
                try:
                    with Image.open(full_img_path) as img:
                        img_width, img_height = img.size
                        if img_width <= 0 or img_height <= 0:
                            print(f"Warning: Invalid image dimensions: {full_img_path}")
                            skipped_images += 1
                            continue
                except Exception as e:
                    print(f"Warning: Cannot open image {full_img_path}: {e}")
                    skipped_images += 1
                    continue
                
                # Get directory and filename
                img_dir, img_filename = os.path.split(full_img_path)
                img_name = os.path.splitext(img_filename)[0]
                
                # Read gt_data.txt to get mask count
                gt_data_path = os.path.join(img_dir, f"{img_name}.gt_data.txt")
                if os.path.exists(gt_data_path):
                    try:
                        with open(gt_data_path, 'r') as gt_file:
                            mask_count = len(gt_file.readlines())
                    except Exception as e:
                        print(f"Warning: Error reading gt_data file {gt_data_path}: {e}")
                        mask_count = 0
                else:
                    # If gt_data.txt doesn't exist, try to determine mask count by finding files
                    mask_count = 0
                    while os.path.exists(os.path.join(img_dir, f"{img_name}.mask{mask_count:02d}.png")):
                        mask_count += 1
                
                # Store all masks for this image
                mask_paths = []
                for mask_idx in range(mask_count):
                    mask_path = os.path.join(img_dir, f"{img_name}.mask{mask_idx:02d}.png")
                    if os.path.exists(mask_path):
                        # Verify mask is a valid image
                        try:
                            with Image.open(mask_path) as mask:
                                pass  # Just testing if it opens
                            mask_paths.append(mask_path)
                        except Exception as e:
                            print(f"Warning: Invalid mask file {mask_path}: {e}")
                
                if mask_paths:  # Only add images that have at least one mask
                    self.image_data.append({
                        'image_path': full_img_path,
                        'mask_paths': mask_paths,
                        'image_name': img_name
                    })
                    valid_images += 1
                else:
                    skipped_images += 1
                    
            except Exception as e:
                print(f"Error processing image path {img_path}: {e}")
                skipped_images += 1
                
        print(f"Found {valid_images} valid images with masks, skipped {skipped_images} invalid entries")
        
        print(f"Processed {len(self.image_data)} images with masks")

    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image and mask paths
        img_data = self.image_data[idx]
        img_path = img_data['image_path']
        mask_paths = img_data['mask_paths']
        img_name = img_data['image_name']
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Always resize image to the same dimensions
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            
            # Create a combined mask by loading and merging all masks
            combined_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
            for mask_path in mask_paths:
                try:
                    mask = Image.open(mask_path).convert('L')
                    # Always resize mask to the same dimensions
                    mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
                    # Convert to numpy and normalize to 0-1
                    mask_np = np.array(mask) / 255.0
                    # Combine masks using maximum (any logo is a positive)
                    combined_mask = np.maximum(combined_mask, mask_np)
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
                    # Continue with other masks if one fails
                    continue
            
            # Convert to tensors
            if self.transform:
                # Apply transformations if specified
                image = self.transform(image)
            else:
                # Default conversion to tensor
                to_tensor = transforms.ToTensor()
                image = to_tensor(image)
            
            # Ensure mask has the correct dimensions - critical for batching
            combined_mask = torch.from_numpy(combined_mask).float().unsqueeze(0)  # Add channel dimension
            
            # Verify tensor shapes before returning
            assert image.shape[1:] == torch.Size([self.image_size, self.image_size]), f"Image shape {image.shape} is incorrect"
            assert combined_mask.shape[1:] == torch.Size([self.image_size, self.image_size]), f"Mask shape {combined_mask.shape} is incorrect"
        
        except Exception as e:
            print(f"Error processing item {idx}, image: {img_path}: {e}")
            # Create empty tensors with correct shapes as fallback
            image = torch.zeros(3, self.image_size, self.image_size)
            combined_mask = torch.zeros(1, self.image_size, self.image_size)
        
        output = {
            'image': image,                # For main model input
            'mask': combined_mask,         # Combined ground truth mask
            'image_path': img_path,        # Original image path
            'mask_paths': mask_paths,      # Original mask paths
            'image_name': img_name,        # Image name without extension
            'num_masks': len(mask_paths)   # Number of masks combined
        }
        
        return output

def custom_collate_fn(batch):
    """
    Custom collate function to handle potential issues with batch sizes
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched samples with consistent sizes
    """
    # Filter out problematic samples (where any tensor is None)
    filtered_batch = []
    for sample in batch:
        if sample is None:
            continue
        if 'image' not in sample or 'mask' not in sample:
            continue
        if sample['image'] is None or sample['mask'] is None:
            continue
        if not isinstance(sample['image'], torch.Tensor) or not isinstance(sample['mask'], torch.Tensor):
            continue
            
        filtered_batch.append(sample)
    
    # Return empty batch if all samples were filtered out
    if len(filtered_batch) == 0:
        return {
            'image': torch.tensor([]),
            'mask': torch.tensor([]),
            'image_path': [],
            'mask_paths': [],
            'image_name': [],
            'num_masks': []
        }
    
    # Verify all tensors have the same shape
    img_shapes = set(sample['image'].shape for sample in filtered_batch)
    mask_shapes = set(sample['mask'].shape for sample in filtered_batch)
    
    if len(img_shapes) > 1 or len(mask_shapes) > 1:
        # If shapes vary, resize tensors to the first sample's shape
        ref_img_shape = filtered_batch[0]['image'].shape
        ref_mask_shape = filtered_batch[0]['mask'].shape
        
        for i in range(len(filtered_batch)):
            if filtered_batch[i]['image'].shape != ref_img_shape:
                filtered_batch[i]['image'] = torch.nn.functional.interpolate(
                    filtered_batch[i]['image'].unsqueeze(0),
                    size=(ref_img_shape[1], ref_img_shape[2]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
            if filtered_batch[i]['mask'].shape != ref_mask_shape:
                filtered_batch[i]['mask'] = torch.nn.functional.interpolate(
                    filtered_batch[i]['mask'].unsqueeze(0),
                    size=(ref_mask_shape[1], ref_mask_shape[2]),
                    mode='nearest'
                ).squeeze(0)
    
    # Stack tensor fields
    image_batch = torch.stack([sample['image'] for sample in filtered_batch])
    mask_batch = torch.stack([sample['mask'] for sample in filtered_batch])
    
    # Collect non-tensor fields
    image_paths = [sample['image_path'] for sample in filtered_batch]
    mask_paths = [sample['mask_paths'] for sample in filtered_batch]
    image_names = [sample['image_name'] for sample in filtered_batch]
    num_masks = [sample['num_masks'] for sample in filtered_batch]
    
    return {
        'image': image_batch,
        'mask': mask_batch,
        'image_path': image_paths,
        'mask_paths': mask_paths,
        'image_name': image_names,
        'num_masks': num_masks
    }

def get_logo_dataloader(root_dir, filelist_path, batch_size=4, image_size=512, num_workers=4, shuffle=True):
    """
    Create and return a dataloader for the logo dataset with combined masks
    """
    # Define transformations
    transform = transforms.Compose([
        # We don't need to resize here as it's handled in the dataset
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Create dataset
    dataset = LogoDatasetCombined(
        root_dir=root_dir,
        filelist_path=filelist_path,
        image_size=image_size,
        transform=transform
    )
    
    # Create dataloader with proper error handling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up the transfer of data to GPU
        drop_last=True,   # Drop the last incomplete batch if dataset size is not divisible by batch_size
        collate_fn=custom_collate_fn  # Use custom collate function to handle size mismatches
    )
    
    return dataloader, dataset