import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid

def visualize_batch(batch, num_samples=4, denormalize=True):
    """
    Visualize a batch of images and masks
    
    Args:
        batch: Dictionary containing 'image' and 'mask' tensors
        num_samples: Number of samples to visualize
        denormalize: Whether to denormalize the images
        
    Returns:
        fig: Matplotlib figure
    """
    images = batch['image'][:num_samples].cpu()
    masks = batch['mask'][:num_samples].cpu()
    
    # Denormalize images if needed
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Display image
        axes[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 0].axis('off')
        
        # Display mask
        axes[i, 1].imshow(masks[i].squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title(f"Mask {i+1}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_predictions(model, dataloader, device, num_samples=4):
    """
    Visualize model predictions on a batch of data
    
    Args:
        model: Trained model
        dataloader: DataLoader object
        device: Device to run the model on
        num_samples: Number of samples to visualize
        
    Returns:
        fig: Matplotlib figure
    """
    model.eval()
    batch = next(iter(dataloader))
    
    images = batch['image'][:num_samples].to(device)
    masks = batch['mask'][:num_samples].cpu()
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
    
    # Move predictions to CPU
    preds = preds[:num_samples].cpu()
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images.cpu() * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Display image
        axes[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 0].axis('off')
        
        # Display ground truth mask
        axes[i, 1].imshow(masks[i].squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title(f"Ground Truth {i+1}")
        axes[i, 1].axis('off')
        
        # Display prediction
        axes[i, 2].imshow(preds[i].squeeze().numpy(), cmap='gray')
        axes[i, 2].set_title(f"Prediction {i+1}")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig

def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate the model on a dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader object
        device: Device to run the model on
        threshold: Threshold for binary prediction
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > threshold).float()
            
            # Calculate metrics
            for i in range(preds.size(0)):
                pred = preds[i].view(-1)
                mask = masks[i].view(-1)
                
                # Calculate Dice
                intersection = (pred * mask).sum()
                dice = (2.0 * intersection) / (pred.sum() + mask.sum() + 1e-6)
                dice_scores.append(dice.item())
                
                # Calculate IoU
                union = pred.sum() + mask.sum() - intersection
                iou = intersection / (union + 1e-6)
                iou_scores.append(iou.item())
    
    # Calculate mean metrics
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    
    return {
        'dice': mean_dice,
        'iou': mean_iou,
        'dice_scores': dice_scores,
        'iou_scores': iou_scores
    }

def predict_single_image(model, image_path, device, image_size=512):
    """
    Make a prediction for a single image
    
    Args:
        model: Trained model
        image_path: Path to the image
        device: Device to run the model on
        image_size: Size to resize the image to
        
    Returns:
        image: Original image (denormalized)
        pred: Prediction mask
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output)
    
    # Move to CPU
    image_tensor = image_tensor.cpu()
    pred = pred.cpu()
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_tensor = image_tensor * std + mean
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    return image_tensor.squeeze(), pred.squeeze()

def overlay_mask(image, mask, alpha=0.5, colormap='jet'):
    """
    Overlay a mask on an image
    
    Args:
        image: Image tensor [C, H, W]
        mask: Mask tensor [H, W] or [1, H, W]
        alpha: Transparency of the overlay
        colormap: Colormap to use for the mask
        
    Returns:
        overlaid: Overlaid image
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    
    # Convert to numpy
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()
    
    # Create colormap
    cmap = plt.get_cmap(colormap)
    mask_colored = cmap(mask_np)[:, :, :3]  # Remove alpha channel
    
    # Overlay
    overlaid = image_np * (1 - alpha) + mask_colored * alpha
    overlaid = np.clip(overlaid, 0, 1)
    
    return overlaid

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    axes[0].plot(history['epochs'], history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['epochs'], history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    
    # Plot Dice score
    axes[1].plot(history['epochs'], history['train_dice'], label='Train Dice')
    if 'val_dice' in history:
        axes[1].plot(history['epochs'], history['val_dice'], label='Val Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training and Validation Dice Score')
    axes[1].legend()
    
    plt.tight_layout()
    return fig